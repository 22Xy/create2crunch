import time
import sys
from collections import defaultdict

def main():
    filepath = 'efficient_addresses.txt'
    try:
        with open(filepath, 'r') as file:
            # Initialize by seeking to the beginning of the file
            file.seek(0, 0)
            last_pos = file.tell()
            
            start_time = time.time()
            total_addresses = 0
            total_scores = 0
            score_distribution = defaultdict(int)
            
            print(f"Starting monitoring of '{filepath}'...\n")
            
            while True:
                # Move to the last known position
                file.seek(last_pos)
                new_lines = file.readlines()
                last_pos = file.tell()
                
                for line in new_lines:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    # Expected format:
                    # {salt} => {address} => Score: {score}
                    try:
                        salt_part, address_part, score_part = line.split('=>')
                        salt = salt_part.strip()
                        address = address_part.strip()
                        # Extract score
                        score_str = score_part.strip()
                        score = int(score_str.split(':')[-1].strip())
                        
                        # Update statistics
                        total_addresses += 1
                        total_scores += score
                        score_distribution[score] += 1
                        
                        # Display the newly found address
                        # print(f"{salt} => {address} => Score: {score}")
                    except Exception as e:
                        print(f"Failed to parse line: \"{line}\". Error: {e}", file=sys.stderr)
                        continue  # Skip malformed lines
                
                # Calculate runtime and submission rate
                elapsed_time = time.time() - start_time
                minutes = elapsed_time / 60 if elapsed_time > 0 else 0
                rate = total_addresses / minutes if minutes > 0 else 0
                
                # Display summary statistics
                print("\n=== Summary ===")
                print(f"Runtime: {minutes:.2f} minutes")
                print(f"Total addresses found: {total_addresses}")
                print(f"Total scores accumulated: {total_scores}")
                print(f"Submission rate: {rate:.2f} addresses per minute")
                print("Score distribution:")
                for score in sorted(score_distribution.keys()):
                    count = score_distribution[score]
                    print(f"  Score {score}: {count} address(es)")
                print("================\n")
                
                # Sleep before the next check
                time.sleep(3600)  # Check every 1 hr
    except FileNotFoundError:
        print(f"Error: File \"{filepath}\" not found. Please ensure that 'create2crunch' is generating it.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
