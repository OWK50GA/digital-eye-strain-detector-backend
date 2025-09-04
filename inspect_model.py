# inspect_model.py
import torch

# Load the model checkpoint
checkpoint = torch.load('resnet_lstm_best_with_threshold.pth', map_location='cpu')

# Print the state_dict keys to see the actual architecture
print("State dict keys:")
for key in checkpoint['model_state_dict'].keys():
    print(f"  {key}")

# Print model info from checkpoint
print("\nCheckpoint info:")
for key, value in checkpoint.items():
    if key != 'model_state_dict':
        print(f"  {key}: {value}")

# Try to determine the architecture from the keys
print("\nArchitecture analysis:")
lstm_keys = [k for k in checkpoint['model_state_dict'].keys() if 'lstm' in k]
attn_keys = [k for k in checkpoint['model_state_dict'].keys() if 'attn' in k]
classifier_keys = [k for k in checkpoint['model_state_dict'].keys() if 'classifier' in k]

print(f"LSTM layers: {lstm_keys}")
print(f"Attention layers: {attn_keys}")
print(f"Classifier layers: {classifier_keys}")