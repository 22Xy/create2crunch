use rustc_hash::FxHashMap;

/// Struct to handle reward calculations based on Uniswap v4 criteria.
pub struct Reward {
    reward: FxHashMap<usize, &'static str>,
}

impl Reward {
    /// Initializes the reward mapping based on Uniswap v4 scoring criteria.
    pub fn new() -> Self {
        let reward = FxHashMap::from_iter([
            // Example mappings; adjust as needed.
            // The key can be a unique identifier based on scoring criteria.
            // For simplicity, you might map total points to a string representation.
            (10, "10"),
            (20, "20"),
            (40, "40"),
            (50, "50"),
            // Add more mappings as per your scoring logic.
        ]);
        Reward { reward }
    }

    /// Calculates the score for a given Ethereum address.
    /// Returns `None` if the address does not meet the first non-zero nibble criteria.
    pub fn calculate_score(&self, address: &str) -> Option<usize> {
        let mut score = 0;

        // Ensure the address is lowercase and strip the '0x' prefix.
        let addr = address.trim_start_matches("0x").to_lowercase();

        // Find the first non-zero nibble
        let first_non_zero = addr.chars().find(|&c| c != '0')?;
        if first_non_zero != '4' {
            // Address does not meet the first non-zero nibble requirement
            return None;
        }

        // Add 10 points for each leading '0' nibble before the first non-zero nibble
        for c in addr.chars() {
            if c == '0' {
                score += 10;
            } else {
                break;
            }
        }

        // Check if the address starts with four consecutive '4's.
        if addr.starts_with("4444") {
            score += 40;

            // Check the first nibble after the four '4's.
            if addr.chars().nth(4) != Some('4') {
                score += 20;
            }
        }

        // Check if the last four nibbles are all '4's.
        if addr.ends_with("4444") {
            score += 20;
        }

        // Add 1 point for each '4' elsewhere in the address.
        for c in addr.chars().skip(1).take(addr.len() - 1) {
            if c == '4' {
                score += 1;
            }
        }

        if score == 0 {
            None
        } else {
            Some(score)
        }
    }

    /// Retrieves the reward based on the calculated score.
    pub fn get(&self, score: &usize) -> Option<&'static str> {
        self.reward.get(score).copied()
    }
}
