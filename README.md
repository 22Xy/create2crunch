## Salt Miner for Uniswap v4

For the detailed rules and criteria of the challenge, see https://blog.uniswap.org/uniswap-v4-address-mining-challenge

To run it locally, first copy `.env.example`

```bash
cp .env.example .env
```

Replace `CALLER` with the Ethereum address executing the submission

```
CALLER="<YOUR_PUB_KEY>"
```

(Optional) Change the `SCORE_MIN_THRESHOLD` to the minimum score to consider. i.e., only salt with a score greater than the threshold will be logged and recorded.

```
SCORE_MIN_THRESHOLD="10"
```

Build the salt miner optimally

```bash
cargo build --release
```

Run the salt miner

```bash
cargo run --release
```

Example Output (CPU mining)

```bash
Total runtime: 0:00:20.080167055130005 (281474976710655 cycles)
Rate: 14017561.50 million attempts per second
Total found this run: 2
Score thresholds: min=90, max=100000

0xbd24513ed63130e883553105c1893d540372adc60a7fb862b875f3d3e7000000 => 0x00044446648fd9f03b834afcb4f63c0192ae152d => Score: 97

0xbd24513ed63130e883553105c1893d540372adc60a7fb862b875179514020000 => 0x00044449b22d31e1cbd3e2de90b33f0bda989014 => Score: 95
```

Example Output (GPU mining)

```bash
Total runtime: 0:00:7.089572906494141 (42 cycles)
Rate: 396.92 million attempts per second
Total found this run: 2
Current search space: 090836f7xxxxxxxx859388b1
Score thresholds: min=0, max=100000

0xbd24513ed63130e883553105c1893d540372adc6c0087372e0ffff03d43ec597 => 0x4d4a91e0197b9e6d0bd32ef6d76d460f40b7a5e2 => Score: 4

0xbd24513ed63130e883553105c1893d540372adc6188043c760ffff03545a2da0 => 0x4d2c88a006228ba4df9389c79b67acadf2f5eee4 => Score: 3
```

(Optional) Run a monitoring tool in another tab

```bash
python3 analysis.py
```

Example Output (Monitoring)

```bash
=== Summary ===
Runtime: 0.00 minutes
Total addresses found: 358193
Total scores accumulated: 2966298
Submission rate: 120058202.90 addresses per minute
Score distribution:
  Score 1: 17005 address(es)
  Score 2: 44523 address(es)
  Score 3: 57202 address(es)
  Score 4: 46595 address(es)
  Score 5: 28061 address(es)
  Score 6: 12940 address(es)
  Score 7: 5015 address(es)
  Score 8: 1546 address(es)
  Score 9: 411 address(es)
  Score 10: 91 address(es)
  Score 11: 11182 address(es)
  Score 12: 27672 address(es)
  Score 13: 33762 address(es)
  Score 14: 27234 address(es)
  Score 15: 15800 address(es)
  Score 16: 7109 address(es)
  Score 17: 2598 address(es)
  Score 18: 784 address(es)
  Score 19: 198 address(es)
  Score 20: 49 address(es)
  Score 21: 1391 address(es)
  Score 22: 3275 address(es)
  Score 23: 3949 address(es)
  Score 24: 3065 address(es)
  Score 25: 1709 address(es)
  Score 26: 779 address(es)
  Score 27: 284 address(es)
  Score 28: 84 address(es)
  Score 29: 26 address(es)
  Score 30: 7 address(es)
  Score 31: 139 address(es)
  Score 32: 348 address(es)
  Score 33: 422 address(es)
  Score 34: 310 address(es)
  Score 35: 165 address(es)
  Score 36: 70 address(es)
  Score 37: 28 address(es)
  Score 38: 9 address(es)
  Score 40: 3 address(es)
  Score 41: 10 address(es)
  Score 42: 26 address(es)
  Score 43: 30 address(es)
  Score 44: 27 address(es)
  Score 45: 33 address(es)
  Score 46: 38 address(es)
  Score 47: 19 address(es)
  Score 48: 22 address(es)
  Score 49: 8 address(es)
  Score 50: 5 address(es)
  Score 51: 3 address(es)
  Score 53: 2 address(es)
  Score 54: 2 address(es)
  Score 55: 3 address(es)
  Score 56: 2 address(es)
  Score 58: 1 address(es)
  Score 59: 1 address(es)
  Score 63: 62 address(es)
  Score 64: 275 address(es)
  Score 65: 525 address(es)
  Score 66: 500 address(es)
  Score 67: 309 address(es)
  Score 68: 156 address(es)
  Score 69: 65 address(es)
  Score 70: 20 address(es)
  Score 71: 4 address(es)
  Score 72: 4 address(es)
  Score 74: 10 address(es)
  Score 75: 26 address(es)
  Score 76: 15 address(es)
  Score 77: 15 address(es)
  Score 78: 10 address(es)
  Score 79: 9 address(es)
  Score 80: 1 address(es)
  Score 81: 6 address(es)
  Score 82: 1 address(es)
  Score 84: 9 address(es)
  Score 85: 24 address(es)
  Score 86: 14 address(es)
  Score 87: 13 address(es)
  Score 88: 11 address(es)
  Score 89: 10 address(es)
  Score 90: 1 address(es)
  Score 91: 1 address(es)
  Score 94: 2 address(es)
  Score 95: 4 address(es)
  Score 96: 1 address(es)
  Score 97: 3 address(es)
  Score 105: 2 address(es)
  Score 106: 1 address(es)
  Score 108: 2 address(es)
  Score 115: 2 address(es)
  Score 116: 5 address(es)
  Score 117: 2 address(es)
  Score 118: 1 address(es)
================
```

## create2crunch

> A Rust program for finding salts that create gas-efficient Ethereum addresses via CREATE2.

Provide three arguments: a factory address (or contract that will call CREATE2), a caller address (for factory addresses that require it as a protection against frontrunning), and the keccak-256 hash of the initialization code of the contract that the factory will deploy.
(The example below references the `Create2Factory`'s address on one of the 21 chains where it has been deployed to.)

Live `Create2Factory` contracts can be found [here](https://blockscan.com/address/0x0000000000ffe8b47b3e2130213b802212439497).

```sh
$ git clone https://github.com/0age/create2crunch
$ cd create2crunch
$ export FACTORY="0x0000000000ffe8b47b3e2130213b802212439497"
$ export CALLER="<YOUR_DEPLOYER_ADDRESS_OF_CHOICE_GOES_HERE>"
$ export INIT_CODE_HASH="<HASH_OF_YOUR_CONTRACT_INIT_CODE_GOES_HERE>"
$ cargo run --release $FACTORY $CALLER $INIT_CODE_HASH
```

For each efficient address found, the salt, resultant addresses, and value _(i.e. approximate rarity)_ will be written to `efficient_addresses.txt`. Verify that one of the salts actually results in the intended address before getting in too deep - ideally, the CREATE2 factory will have a view method for checking what address you'll get for submitting a particular salt. Be sure not to change the factory address or the init code without first removing any existing data to prevent the two salt types from becoming commingled. There's also a _very_ simple monitoring tool available if you run `$python3 analysis.py` in another tab.

This tool was originally built for use with [`Pr000xy`](https://github.com/0age/Pr000xy), including with [`Create2Factory`](https://github.com/0age/Pr000xy/blob/master/contracts/Create2Factory.sol) directly.

There is also an experimental OpenCL feature that can be used to search for addresses using a GPU. To give it a try, include a fourth parameter specifying the device ID to use, and optionally a fifth and sixth parameter to filter returned results by a threshold based on leading zero bytes and total zero bytes, respectively. By way of example, to perform the same search as above, but using OpenCL device 2 and only returning results that create addresses with at least four leading zeroes or six total zeroes, use `$ cargo run --release $FACTORY $CALLER $INIT_CODE_HASH 2 4 6` (you'll also probably want to try tweaking the `WORK_SIZE` parameter in `src/lib.rs`).

PRs welcome!
