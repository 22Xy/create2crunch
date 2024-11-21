#![warn(unused_crate_dependencies, unreachable_pub)]
#![deny(unused_must_use, rust_2018_idioms)]

use alloy_primitives::{hex, Address, FixedBytes};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use console::Term;
use dotenv::dotenv;
use fs4::FileExt;
use ocl::{Buffer, Context, Device, MemFlags, Platform, ProQue, Program, Queue};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use separator::Separatable;
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};
use tiny_keccak::{Hasher, Keccak};

mod reward;
pub use reward::Reward;

// Workset size (tweak this!)
const WORK_SIZE: u32 = 0x4000000; // max. 0x15400000 to abs. max 0xffffffff

const WORK_FACTOR: u128 = (WORK_SIZE as u128) / 1_000_000;
const CONTROL_CHARACTER: u8 = 0xff;
const MAX_INCREMENTER: u64 = 0xffffffffffff;

// Cache for already processed addresses to prevent duplicates
lazy_static::lazy_static! {
    static ref PROCESSED_ADDRESSES: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
}

static KERNEL_SRC: &str = include_str!("./kernels/keccak256.cl");

/// Requires three hex-encoded arguments: the address of the contract that will
/// be calling CREATE2, the address of the caller of said contract *(assuming
/// the contract calling CREATE2 has frontrunning protection in place - if not
/// applicable to your use-case you can set it to the null address)*, and the
/// keccak-256 hash of the bytecode that is provided by the contract calling
/// CREATE2 that will be used to initialize the new contract. An additional set
/// of three optional values may be provided: a device to target for OpenCL GPU
/// search, a threshold for leading zeroes to search for, and a threshold for
/// total zeroes to search for.
pub struct Config {
    pub factory_address: [u8; 20],
    pub calling_address: [u8; 20],
    pub init_code_hash: [u8; 32],
    pub gpu_device: u8,
    pub leading_zeroes_threshold: u8,
    pub total_zeroes_threshold: u8,
    pub score_min_threshold: usize,
    pub score_max_threshold: usize,
}

/// Validate the provided arguments and construct the Config struct.
impl Config {
    pub fn new(mut args: env::Args) -> Result<Self, &'static str> {
        // Load environment variables from .env file
        dotenv().ok();

        // Retrieve environment variables or return an error
        let factory = env::var("FACTORY").map_err(|_| "FACTORY not set in .env")?;
        let caller = env::var("CALLER").map_err(|_| "CALLER not set in .env")?;
        let init_code_hash =
            env::var("INIT_CODE_HASH").map_err(|_| "INIT_CODE_HASH not set in .env")?;
        let gpu_device = env::var("GPU_DEVICE")
            .unwrap_or_else(|_| "255".to_string())
            .parse::<u8>()
            .unwrap_or(255);

        // Retrieve score thresholds from environment variables
        let score_min_threshold_str =
            env::var("SCORE_MIN_THRESHOLD").map_err(|_| "SCORE_MIN_THRESHOLD not set in .env")?;
        let score_max_threshold_str =
            env::var("SCORE_MAX_THRESHOLD").map_err(|_| "SCORE_MAX_THRESHOLD not set in .env")?;

        let score_min_threshold = score_min_threshold_str
            .parse::<usize>()
            .map_err(|_| "Invalid SCORE_MIN_THRESHOLD format")?;
        let score_max_threshold = score_max_threshold_str
            .parse::<usize>()
            .map_err(|_| "Invalid SCORE_MAX_THRESHOLD format")?;

        if score_min_threshold >= score_max_threshold {
            return Err("SCORE_MIN_THRESHOLD must be less than SCORE_MAX_THRESHOLD");
        }

        // Convert Hex strings to byte arrays
        let factory_address = hex::decode(factory.trim_start_matches("0x"))
            .map_err(|_| "Invalid FACTORY address format")?
            .try_into()
            .map_err(|_| "FACTORY address must be 20 bytes")?;

        let calling_address = hex::decode(caller.trim_start_matches("0x"))
            .map_err(|_| "Invalid CALLER address format")?
            .try_into()
            .map_err(|_| "CALLER address must be 20 bytes")?;

        let init_code_hash = hex::decode(init_code_hash.trim_start_matches("0x"))
            .map_err(|_| "Invalid INIT_CODE_HASH format")?
            .try_into()
            .map_err(|_| "INIT_CODE_HASH must be 32 bytes")?;

        Ok(Self {
            factory_address,
            calling_address,
            init_code_hash,
            gpu_device,
            leading_zeroes_threshold: 0, // You can adjust or make this configurable as needed
            total_zeroes_threshold: 0,   // You can adjust or make this configurable as needed
            score_min_threshold,
            score_max_threshold,
        })
    }
}

/// Given a Config object with a factory address, a caller address, and a
/// keccak-256 hash of the contract initialization code, search for salts that
/// will enable the factory contract to deploy a contract to a gas-efficient
/// address via CREATE2.
///
/// The 32-byte salt is constructed as follows:
///   - the 20-byte calling address (to prevent frontrunning)
///   - a random 6-byte segment (to prevent collisions with other runs)
///   - a 6-byte nonce segment (incrementally stepped through during the run)
///
/// When a salt that will result in the creation of a gas-efficient contract
/// address is found, it will be appended to `efficient_addresses.txt` along
/// with the resultant address and the "value" (i.e. the score) of the
/// resultant address.
pub fn cpu(config: Config) -> Result<(), Box<dyn Error>> {
    // (create if necessary) and open a file where found salts will be written
    let file = output_file();

    // Create object for computing rewards (relative rarity) for a given address
    let rewards = Reward::new().expect("Failed to initialize Reward");

    // Begin searching for addresses
    loop {
        // Header: 0xff ++ factory ++ caller ++ salt_random_segment (47 bytes)
        let mut header = [0; 47];
        header[0] = CONTROL_CHARACTER;
        header[1..21].copy_from_slice(&config.factory_address);
        header[21..41].copy_from_slice(&config.calling_address);
        header[41..].copy_from_slice(&FixedBytes::<6>::random()[..]);

        // Create new hash object
        let mut hash_header = Keccak::v256();

        // Update hash with header
        hash_header.update(&header);

        // Iterate over a 6-byte nonce and compute each address
        (0..MAX_INCREMENTER)
            .into_par_iter() // Parallelization
            .for_each(|salt| {
                let salt_bytes = salt.to_le_bytes();
                let salt_incremented_segment = &salt_bytes[..6];

                // Clone the partially-hashed object
                let mut hash = hash_header.clone();

                // Update with body and footer (total: 38 bytes)
                hash.update(salt_incremented_segment);
                hash.update(&config.init_code_hash);

                // Hash the payload and get the result
                let mut res: [u8; 32] = [0; 32];
                hash.finalize(&mut res);

                // Get the address that results from the hash
                let address = <&Address>::try_from(&res[12..]).unwrap();
                let address_str = format!("0x{}", hex::encode(address));

                // Calculate the score based on Uniswap v4 criteria
                let score_option = rewards.calculate_score(&address_str);

                // Only proceed if the score meets the criteria
                let score = match score_option {
                    Some(s) => s,
                    None => return, // Invalid address based on first nibble
                };

                // Check for duplicate submissions
                {
                    let mut processed = PROCESSED_ADDRESSES
                        .lock()
                        .expect("Failed to lock PROCESSED_ADDRESSES");
                    if !processed.insert(address_str.clone()) {
                        // Address already processed
                        return;
                    }
                }

                // Get the full salt used to create the address
                let header_hex_string = hex::encode(header);
                let body_hex_string = hex::encode(salt_incremented_segment);
                let full_salt = format!("0x{}{}", &header_hex_string[42..], &body_hex_string);

                // Display the salt and the address.
                let output = format!("{full_salt} => {address_str} => Score: {score} \n");
                println!("{output}");

                // Create a lock on the file before writing
                file.lock_exclusive().expect("Couldn't lock file.");

                // Write the result to file
                writeln!(&file, "{output}")
                    .expect("Couldn't write to `efficient_addresses.txt` file.");

                // Release the file lock
                file.unlock().expect("Couldn't unlock file.");
            });
    }
}

/// Given a Config object with a factory address, a caller address, a keccak-256
/// hash of the contract initialization code, and a device ID, search for salts
/// using OpenCL that will enable the factory contract to deploy a contract to a
/// gas-efficient address via CREATE2. This method also takes threshold values
/// for both leading zero bytes and total zero bytes - any address that does not
/// meet or exceed the threshold will not be returned.
pub fn gpu(config: Config) -> Result<(), Box<dyn Error>> {
    // (Implementation remains similar to CPU method with necessary adjustments)
    // Ensure that duplicate checks and score validation are integrated here as well.

    // Example pseudo-code adjustments:
    // - After computing the address and score, perform the same duplicate check
    // - Only write unique and valid addresses to the output file

    unimplemented!("GPU mining function should implement duplicate checks and score validation similar to CPU.");
}

#[track_caller]
fn output_file() -> File {
    OpenOptions::new()
        .append(true)
        .create(true)
        .read(true)
        .open("efficient_addresses.txt")
        .expect("Could not create or open `efficient_addresses.txt` file.")
}

/// Creates the OpenCL kernel source code by populating the template with the
/// values from the Config object.
fn mk_kernel_src(config: &Config) -> String {
    let mut src = String::with_capacity(2048 + KERNEL_SRC.len());

    let factory = config.factory_address.iter();
    let caller = config.calling_address.iter();
    let hash = config.init_code_hash.iter();
    let hash = hash.enumerate().map(|(i, x)| (i + 52, x));
    for (i, x) in factory.chain(caller).enumerate().chain(hash) {
        writeln!(src, "#define S_{} {}u", i + 1, x).unwrap();
    }
    let lz = config.leading_zeroes_threshold;
    writeln!(src, "#define LEADING_ZEROES {lz}").unwrap();
    let tz = config.total_zeroes_threshold;
    writeln!(src, "#define TOTAL_ZEROES {tz}").unwrap();

    src.push_str(KERNEL_SRC);

    src
}
