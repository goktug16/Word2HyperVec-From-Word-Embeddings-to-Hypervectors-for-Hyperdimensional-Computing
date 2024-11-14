import numpy as np
import json

def create_hypervector_dict(increment):
    total_bits = 10000
    dict_precision = int(np.ceil(-np.log10(increment)))
    
    # Adjusted range: start from -1.5 to 1.5
    start_value, end_value = 0.0, 1.0
    
    # Calculate total number of increments in the new range
    total_increments = int(np.ceil((end_value - start_value) / increment))
    
    # Shuffle all indices to ensure each bit is flipped uniquely across the range
    all_indices = np.arange(total_bits)
    np.random.shuffle(all_indices)
    
    # Initialize the dictionary to store hypervectors
    # Starting with the first value at -1.5
    hypervector_dict = {format(start_value, f'.{dict_precision}f'): np.full(total_bits, -1)}
    
    # Flip bits for each increment in the new range
    for i in range(1, total_increments + 1):
        current_value = round(start_value + i * increment, dict_precision)
        
        # Ensure previous_value is within the defined range
        previous_value = format(max(start_value, current_value - increment), f'.{dict_precision}f')
        new_hypervector = np.copy(hypervector_dict[previous_value])
        
        # Determine the indices to flip for this increment
        indices_to_flip = all_indices[((i - 1) * total_bits) // total_increments : (i * total_bits) // total_increments]
        
        # Flip the bits at the selected indices
        new_hypervector[indices_to_flip] *= -1
        
        # Add the new hypervector to the dictionary
        hypervector_dict[format(current_value, f'.{dict_precision}f')] = new_hypervector

    return hypervector_dict

increment = 0.0001  # Example increment value

hypervector_dict = create_hypervector_dict(increment)

# Convert numpy arrays to lists for JSON serialization
for key in hypervector_dict:
    hypervector_dict[key] = hypervector_dict[key].tolist()

filename = f'hdc_10k_{increment}.json'
with open(filename, 'w') as file:
    json.dump(hypervector_dict, file)

print(f"Hypervector dictionary saved to {filename}")
