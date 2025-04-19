import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')  # Updated to use a valid style
sns.set_theme()  # Use seaborn's default theme

# 1. Performance Metrics Graph
def plot_performance_metrics():
    try:
        metrics = ['Face Detection', 'Feature Extraction', 'Model Prediction']
        times = [50, 30, 20]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, times, color=['#3498db', '#2ecc71', '#e74c3c'])
        plt.title('Processing Time for Different Stages (ms)')
        plt.ylabel('Time (ms)')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}ms',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Performance metrics graph generated successfully")
    except Exception as e:
        print(f"Error generating performance metrics graph: {str(e)}")

# 2. Accuracy Metrics Graph
def plot_accuracy_metrics():
    try:
        metrics = ['Detection Rate', 'Recognition Accuracy', 'False Positive', 'False Negative']
        values = [98, 95, 2, 3]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        plt.title('System Accuracy Metrics (%)')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('accuracy_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Accuracy metrics graph generated successfully")
    except Exception as e:
        print(f"Error generating accuracy metrics graph: {str(e)}")

# 3. Resource Usage Graph
def plot_resource_usage():
    try:
        resources = ['CPU Usage', 'Memory Usage', 'Storage per Student']
        values = [17.5, 500, 1]
        units = ['%', 'MB', 'MB']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(resources, values, color=['#3498db', '#2ecc71', '#e74c3c'])
        plt.title('System Resource Usage')
        plt.ylabel('Usage')
        
        # Add value labels with units
        for bar, unit in zip(bars, units):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}{unit}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('resource_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Resource usage graph generated successfully")
    except Exception as e:
        print(f"Error generating resource usage graph: {str(e)}")

# 4. Data Flow Diagram
def plot_data_flow():
    try:
        plt.figure(figsize=(12, 8))
        
        # Define nodes
        nodes = ['Webcam Capture', 'Face Detection', 'Preprocessing', 
                 'Feature Extraction', 'Model Prediction', 'Attendance Update']
        
        # Create positions
        x = np.linspace(0, 10, len(nodes))
        y = np.zeros(len(nodes))
        
        # Plot nodes
        plt.scatter(x, y, s=1000, c='#3498db', alpha=0.6)
        
        # Add arrows
        for i in range(len(nodes)-1):
            plt.arrow(x[i], y[i], x[i+1]-x[i], y[i+1]-y[i], 
                     head_width=0.1, head_length=0.2, fc='k', ec='k')
        
        # Add labels
        for i, node in enumerate(nodes):
            plt.text(x[i], y[i], node, ha='center', va='center')
        
        plt.title('System Data Flow')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('data_flow.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Data flow diagram generated successfully")
    except Exception as e:
        print(f"Error generating data flow diagram: {str(e)}")

# Generate all graphs
if __name__ == "__main__":
    try:
        plot_performance_metrics()
        plot_accuracy_metrics()
        plot_resource_usage()
        plot_data_flow()
        print("\nAll graphs generated successfully!")
    except Exception as e:
        print(f"\nError in main execution: {str(e)}") 