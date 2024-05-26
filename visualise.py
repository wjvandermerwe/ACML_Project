import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Load TensorBoard logs
log_path = 'runs/tmodel/tmodel'
ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()

# Function to apply moving average smoothing
def smooth_data(data, window_size=10):
    return data.rolling(window=window_size, min_periods=1).mean()

# Extract and plot each metric individually
for tag in ea.Tags()['scalars']:
    # Extract the data for the current metric
    metrics = ea.Scalars(tag)
    df = pd.DataFrame([(x.step, x.value) for x in metrics], columns=['Step', 'Value'])

    # Check if DataFrame is empty
    if df.empty:
        print(f"No data for {tag}")
        continue

    # Apply smoothing
    df['Smoothed Value'] = smooth_data(df['Value'])

    # Plot the metric
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='Step', y='Smoothed Value')
    plt.title(f'{tag} Over Training Steps (Smoothed)')
    plt.xlabel('Training Steps')
    plt.ylabel('Smoothed Value')
    plt.grid(True)
    
    # Save the plot as an image
    image_filename = f'{tag.replace("/", "_")}_smoothed.png'  # Replace characters that might not be valid in file names
    plt.savefig(image_filename)
    plt.close()  # Close the plot to free up memory
    print(f"Saved plot for {tag} as {image_filename}")