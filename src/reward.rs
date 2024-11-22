use std::env;

/// Struct to handle reward calculations based on Uniswap v4 criteria.
/// Full details here: https://blog.uniswap.org/uniswap-v4-address-mining-challenge
pub struct Reward {
    min_threshold: usize,
    max_threshold: usize,
}

impl Reward {
    /// Initializes the Reward struct with thresholds from environment variables.
    pub fn new() -> Result<Self, &'static str> {
        // Retrieve threshold values from environment variables
        let min_threshold_str =
            env::var("SCORE_MIN_THRESHOLD").map_err(|_| "SCORE_MIN_THRESHOLD not set in .env")?;
        let max_threshold_str =
            env::var("SCORE_MAX_THRESHOLD").map_err(|_| "SCORE_MAX_THRESHOLD not set in .env")?;

        // Parse the threshold strings to usize
        let min_threshold = min_threshold_str
            .parse::<usize>()
            .map_err(|_| "Invalid SCORE_MIN_THRESHOLD format")?;
        let max_threshold = max_threshold_str
            .parse::<usize>()
            .map_err(|_| "Invalid SCORE_MAX_THRESHOLD format")?;

        // Validate that min_threshold is less than max_threshold
        if min_threshold >= max_threshold {
            return Err("SCORE_MIN_THRESHOLD must be less than SCORE_MAX_THRESHOLD");
        }

        Ok(Reward {
            min_threshold,
            max_threshold,
        })
    }

    /// Calculates the score for a given Ethereum address.
    /// Returns `None` if the address does not meet the first non-zero nibble criteria
    /// or if the calculated score is not greater than the minimum threshold.
    pub fn calculate_score(&self, address: &str) -> Option<usize> {
        let mut score = 0; // Initialize score without a base value

        // Ensure the address is lowercase and strip the '0x' prefix.
        let addr = address.trim_start_matches("0x").to_lowercase();

        // Find the first non-zero nibble index
        let first_non_zero_idx = addr.find(|c: char| c != '0')?;
        if addr.chars().nth(first_non_zero_idx)? != '4' {
            // Address does not meet the first non-zero nibble requirement
            return None;
        }

        // Add 10 points for each leading '0' nibble before the first non-zero nibble
        let leading_zeros = addr
            .chars()
            .take(first_non_zero_idx)
            .filter(|&c| c == '0')
            .count();
        score += leading_zeros * 10;

        // Substring starting from the first non-zero nibble
        let remainder = &addr[first_non_zero_idx..];

        // Check if the substring starts with four consecutive '4's.
        if remainder.starts_with("4444") {
            score += 40;

            // Check the first nibble after the four '4's.
            if remainder.chars().nth(4) != Some('4') {
                score += 20;
            }
        }

        // Check if the last four nibbles are all '4's.
        if addr.ends_with("4444") {
            score += 20;
        }

        // Add 1 point for each '4' elsewhere in the address.
        // Exclude leading '0's by starting from first_non_zero_idx
        let additional_fours = addr[first_non_zero_idx..]
            .chars()
            .filter(|&c| c == '4')
            .count();
        score += additional_fours;

        // Ensure the score does not exceed the maximum threshold
        if score > self.max_threshold {
            score = self.max_threshold;
        }

        // Only return the score if it is greater than the minimum threshold
        if score > self.min_threshold {
            Some(score)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_calculate_score() {
        // Set environment variables for testing
        env::set_var("SCORE_MIN_THRESHOLD", "100");
        env::set_var("SCORE_MAX_THRESHOLD", "200");

        let reward = Reward::new().expect("Failed to create Reward instance");

        // Valid addresses with scores > 100
        // Example 1: Leading '0's, starts with '4444', first nibble after '4444' != '4', ends with '4444'
        // Score: (10 * 4) + 40 + 20 + 20 + (number of '4's elsewhere) = 40 + 40 + 20 + 20 + 4 = 124
        assert_eq!(reward.calculate_score("0x00004444abcd4444"), Some(124));

        // Example 2: Leading '0's, starts with '4444', first nibble after '4444' != '4', does not end with '4444'
        // Score: (10 * 3) + 40 + 20 + 0 + (number of '4's elsewhere) = 30 + 40 + 20 + 0 + 3 = 93
        assert_eq!(reward.calculate_score("0x00044444abcd1234"), None);

        // Example 3: No leading '0's, starts with '4444', first nibble after '4444' != '4', ends with '4444'
        // Score: 0 + 40 + 20 + 20 + (number of '4's elsewhere) = 40 + 20 + 20 + 0 = 80
        assert_eq!(reward.calculate_score("0x4444abcd4444"), None);

        // Example 4: Leading '0's, does not start with '4444', but meets other criteria
        // Score: (10 * 5) + 0 + 0 + 20 + (number of '4's elsewhere) = 50 + 0 + 0 + 20 + 4 = 74
        assert_eq!(reward.calculate_score("0x000000abcd4444"), None);

        // Example 5: Leading '0's, starts with '4444', first nibble after '4444' is '4', ends with '4444'
        // Score: (10 * 2) + 40 + 0 + 20 + (number of '4's elsewhere) = 20 + 40 + 0 + 20 + 3 = 83
        assert_eq!(reward.calculate_score("0x0044444abcd4444"), None);

        // Valid address with high score
        // Score: (10 * 6) + 40 + 20 + 20 + (number of '4's elsewhere) = 60 + 40 + 20 + 20 + 4 = 144
        assert_eq!(
            reward.calculate_score("0x0000000044444444abcd4444"),
            Some(144)
        );

        // Valid address with score exactly 101
        // Score: (10 * 5) + 40 + 20 + 0 + (number of '4's elsewhere) = 50 + 40 + 20 + 0 + 1 = 111
        assert_eq!(reward.calculate_score("0x00000abcd4444"), Some(111));

        // Invalid addresses (score <= 100 or first non-zero nibble not '4')
        assert_eq!(reward.calculate_score("0x74444abcd"), None);
        assert_eq!(reward.calculate_score("0x0000abcd"), None);
        assert_eq!(reward.calculate_score("0x0000000000000"), None);
    }
}
