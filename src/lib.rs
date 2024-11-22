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
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
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
    pub fn new() -> Result<Self, &'static str> {
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

    let start_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let found = Arc::new(AtomicU64::new(0));
    let cumulative_nonce = Arc::new(AtomicU64::new(0));
    let found_list = Arc::new(Mutex::new(Vec::new()));

    // Clone Arcs for stats thread
    let stats_found = Arc::clone(&found);
    let stats_nonce = Arc::clone(&cumulative_nonce);
    let stats_list = Arc::clone(&found_list);

    // Spawn statistics thread
    thread::spawn(move || {
        loop {
            // Calculate runtime statistics
            let current_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            let total_runtime = current_time - start_time;
            let total_runtime_hrs = total_runtime as u64 / 3600;
            let total_runtime_mins = (total_runtime as u64 - total_runtime_hrs * 3600) / 60;
            let total_runtime_secs = total_runtime
                - (total_runtime_hrs * 3600) as f64
                - (total_runtime_mins * 60) as f64;

            // Calculate rate
            let rate = if total_runtime > 0.0 {
                stats_nonce.load(Ordering::Relaxed) as f64 / total_runtime / 1_000_000.0
            } else {
                0.0
            };

            // Print statistics
            println!("\x1B[2J\x1B[1;1H"); // Clear screen
            println!(
                "Total runtime: {}:{:02}:{:02} ({} cycles)\n\
                 Rate: {:.2} million attempts per second\n\
                 Total found this run: {}\n\
                 Score thresholds: min={}, max={}\n",
                total_runtime_hrs,
                total_runtime_mins,
                total_runtime_secs,
                stats_nonce.load(Ordering::Relaxed),
                rate,
                stats_found.load(Ordering::Relaxed),
                config.score_min_threshold,
                config.score_max_threshold,
            );

            // Display recently found addresses
            let found_list_guard = stats_list.lock().unwrap();
            let last_10: Vec<String> = found_list_guard.iter().rev().take(10).cloned().collect();
            drop(found_list_guard);

            for entry in last_10.iter().rev() {
                println!("{}", entry);
            }

            // Sleep for a second before next update
            thread::sleep(Duration::from_secs(1));
        }
    });

    // Main mining loop
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

        // Increment cumulative nonce BEFORE the parallel processing
        cumulative_nonce.fetch_add(MAX_INCREMENTER, Ordering::Relaxed);

        // Parallel processing
        (0..MAX_INCREMENTER).into_par_iter().for_each(|salt| {
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
            writeln!(&file, "{output}").expect("Couldn't write to `efficient_addresses.txt` file.");

            // Release the file lock
            file.unlock().expect("Couldn't unlock file.");

            if score_option.is_some() {
                found.fetch_add(1, Ordering::Relaxed);
                found_list.lock().unwrap().push(output.clone());
            }
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
    println!(
        "Setting up experimental OpenCL miner using device {}...",
        config.gpu_device
    );

    // (create if necessary) and open a file where found salts will be written
    let file = output_file();

    // Create object for computing rewards (relative rarity) for a given address
    let rewards = Reward::new().expect("Failed to initialize Reward");

    // track how many addresses have been found and information about them
    let mut found: u64 = 0;
    let mut found_list: Vec<String> = vec![];

    // set up a controller for terminal output
    let term = Term::stdout();

    // set up a platform to use
    let platform = Platform::new(ocl::core::default_platform()?);

    // set up the device to use
    let device = Device::by_idx_wrap(platform, config.gpu_device as usize)?;

    // set up the context to use
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    // set up the program to use
    let program = Program::builder()
        .devices(device)
        .src(mk_kernel_src(&config))
        .build(&context)?;

    // set up the queue to use
    let queue = Queue::new(&context, device, None)?;

    // set up the "proqueue" (or amalgamation of various elements) to use
    let ocl_pq = ProQue::new(context, queue, program, Some(WORK_SIZE));

    // create a random number generator
    let mut rng = thread_rng();

    // determine the start time
    let start_time: f64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // set up variables for tracking performance
    let mut rate: f64 = 0.0;
    let mut cumulative_nonce: u64 = 0;

    // the previous timestamp of printing to the terminal
    let mut previous_time: f64 = 0.0;

    // the last work duration in milliseconds
    let mut work_duration_millis: u64 = 0;

    // begin searching for addresses
    loop {
        // construct the 4-byte message to hash, leaving last 8 of salt empty
        let salt = FixedBytes::<4>::random();

        // build a corresponding buffer for passing the message to the kernel
        let message_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(4)
            .copy_host_slice(&salt[..])
            .build()?;

        // reset nonce & create a buffer to view it in little-endian
        // for more uniformly distributed nonces, we shall initialize it to a random value
        let mut nonce: [u32; 1] = rng.gen();
        let mut view_buf = [0; 8];

        // build a corresponding buffer for passing the nonce to the kernel
        let mut nonce_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(1)
            .copy_host_slice(&nonce)
            .build()?;

        // establish a buffer for nonces that result in desired addresses
        let mut solutions: Vec<u64> = vec![0; 1];
        let solutions_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().write_only())
            .len(1)
            .copy_host_slice(&solutions)
            .build()?;

        // repeatedly enqueue kernel to search for new addresses
        loop {
            // build the kernel and define the type of each buffer
            let kern = ocl_pq
                .kernel_builder("hashMessage")
                .arg_named("message", None::<&Buffer<u8>>)
                .arg_named("nonce", None::<&Buffer<u32>>)
                .arg_named("solutions", None::<&Buffer<u64>>)
                .build()?;

            // set each buffer
            kern.set_arg("message", Some(&message_buffer))?;
            kern.set_arg("nonce", Some(&nonce_buffer))?;
            kern.set_arg("solutions", &solutions_buffer)?;

            // enqueue the kernel
            unsafe { kern.enq()? };

            // calculate the current time
            let mut now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let current_time = now.as_secs() as f64;

            // we don't want to print too fast
            let print_output = current_time - previous_time > 0.99;
            previous_time = current_time;

            // Update terminal display
            if print_output {
                term.clear_screen()?;

                // get the total runtime and parse into hours : minutes : seconds
                let total_runtime = current_time - start_time;
                let total_runtime_hrs = total_runtime as u64 / 3600;
                let total_runtime_mins = (total_runtime as u64 - total_runtime_hrs * 3600) / 60;
                let total_runtime_secs = total_runtime
                    - (total_runtime_hrs * 3600) as f64
                    - (total_runtime_mins * 60) as f64;

                // determine the number of attempts being made per second
                let work_rate: u128 = WORK_FACTOR * cumulative_nonce as u128;
                if total_runtime > 0.0 {
                    rate = 1.0 / total_runtime;
                }

                // fill the buffer for viewing the properly-formatted nonce
                LittleEndian::write_u64(&mut view_buf, (nonce[0] as u64) << 32);

                // calculate the terminal height, defaulting to a height of ten rows
                let height = terminal_size().map(|(_w, Height(h))| h).unwrap_or(10);

                // Display mining statistics
                term.write_line(&format!(
                    "Total runtime: {}:{:02}:{:02} ({} cycles)\n\
                     Rate: {:.2} million attempts per second\n\
                     Total found this run: {}\n\
                     Current search space: {}xxxxxxxx{:08x}\n\
                     Score thresholds: min={}, max={}\n",
                    total_runtime_hrs,
                    total_runtime_mins,
                    total_runtime_secs,
                    cumulative_nonce,
                    work_rate as f64 * rate,
                    found,
                    hex::encode(salt),
                    BigEndian::read_u64(&view_buf),
                    config.score_min_threshold,
                    config.score_max_threshold,
                ))?;

                // Display recently found solutions based on terminal height
                let rows = if height < 7 { 1 } else { height as usize - 6 };
                let last_rows: Vec<String> = found_list.iter().cloned().rev().take(rows).collect();
                let ordered: Vec<String> = last_rows.iter().cloned().rev().collect();
                let recently_found = &ordered.join("\n");
                term.write_line(recently_found)?;
            }

            // increment the cumulative nonce (does not reset after a match)
            cumulative_nonce += 1;

            // record the start time of the work
            let work_start_time_millis = now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000;

            // sleep for 98% of the previous work duration to conserve CPU
            if work_duration_millis != 0 {
                std::thread::sleep(std::time::Duration::from_millis(
                    work_duration_millis * 980 / 1000,
                ));
            }

            // read the solutions from the device
            solutions_buffer.read(&mut solutions).enq()?;

            // record the end time of the work and compute how long the work took
            now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            work_duration_millis = (now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000)
                - work_start_time_millis;

            // Process solutions if found
            if solutions[0] != 0 {
                for &solution in &solutions {
                    if solution == 0 {
                        continue;
                    }

                    let solution = solution.to_le_bytes();

                    // Construct the full message for hashing
                    let mut solution_message = [0; 85];
                    solution_message[0] = CONTROL_CHARACTER;
                    solution_message[1..21].copy_from_slice(&config.factory_address);
                    solution_message[21..41].copy_from_slice(&config.calling_address);
                    solution_message[41..45].copy_from_slice(&salt[..]);
                    solution_message[45..53].copy_from_slice(&solution);
                    solution_message[53..].copy_from_slice(&config.init_code_hash);

                    // Create new hash object and compute the address
                    let mut hash = Keccak::v256();
                    hash.update(&solution_message);
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
                        None => continue, // Skip if score doesn't meet criteria
                    };

                    // Check for duplicate submissions
                    {
                        let mut processed = PROCESSED_ADDRESSES
                            .lock()
                            .expect("Failed to lock PROCESSED_ADDRESSES");
                        if !processed.insert(address_str.clone()) {
                            continue; // Skip if address already processed
                        }
                    }

                    // Get the full salt used to create the address
                    let full_salt = format!(
                        "0x{}{}{}",
                        hex::encode(config.calling_address),
                        hex::encode(salt),
                        hex::encode(solution)
                    );

                    // Format and store the output
                    let output = format!("{full_salt} => {address_str} => Score: {score} \n");
                    found_list.push(output.clone());

                    // Write to file with exclusive lock
                    file.lock_exclusive().expect("Couldn't lock file.");
                    writeln!(&file, "{output}")
                        .expect("Couldn't write to `efficient_addresses.txt` file.");
                    file.unlock().expect("Couldn't unlock file.");

                    found += 1;
                }
                break;
            }

            // if no solution has yet been found, increment the nonce
            nonce[0] += 1;

            // update the nonce buffer with the incremented nonce value
            nonce_buffer = Buffer::builder()
                .queue(ocl_pq.queue().clone())
                .flags(MemFlags::new().read_write())
                .len(1)
                .copy_host_slice(&nonce)
                .build()?;
        }
    }
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
