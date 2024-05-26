import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Load TensorBoard logs
log_path = 'runs/tmodel/tmodel'
ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()

# Extract and plot each metric individually
for tag in ea.Tags()['scalars']:
    # Extract the data for the current metric
    metrics = ea.Scalars(tag)
    df = pd.DataFrame([(x.step, x.value) for x in metrics], columns=['Step', 'Value'])

    # Check if DataFrame is empty
    if df.empty:
        print(f"No data for {tag}")
        continue

    # Plot the metric
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='Step', y='Value')
    plt.title(f'{tag} Over Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Value')
    plt.grid(True)
    
    # Save the plot as an image
    image_filename = f'{tag.replace("/", "_")}.png'  # Replace characters that might not be valid in file names
    plt.savefig(image_filename)
    plt.close()  # Close the plot to free up memory
    print(f"Saved plot for {tag} as {image_filename}")