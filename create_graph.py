import os
import sys
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

def get_connection():
    db_url = "postgresql://postgres:TGBCFSFxiLVUyNZInoIAfClDtkrTwZau@monorail.proxy.rlwy.net:18855/railway"
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    return conn, cursor

def fetch_data():
    conn, cursor = get_connection()
    query = "SELECT * FROM inference_timestep_analysis"
    cursor.execute(query)
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    cursor.close()
    conn.close()
    return pd.DataFrame(data, columns=columns)

def map_model_names(model_name):
    model_mapping = {
        'ADPR_x0': 'ADPR',
        'MB_ADPR_x0': 'MB-ADPR',
        'SAD_x0': 'S-ADPR',
        'S_SAD_x0': 'F-ADPR',
        'LS_SAD_x0': 'LF-ADPR'
    }
    return model_mapping.get(model_name, model_name)

def get_model_color(model_name):
    color_mapping = {
        'ADPR': 'blue',
        'MB-ADPR': 'green',
        'S-ADPR': 'red',
        'F-ADPR': 'orange',
        'LF-ADPR': 'purple'
    }
    return color_mapping.get(model_name, 'black')

def plot_and_save_graphs(data, dataset, timesteps, output_dir):
    metrics = ['precision', 'recall', 'ndcg', 'coverage_item', 'coverage_item_hit']
    model_order = ['ADPR_x0', 'MB_ADPR_x0', 'SAD_x0', 'S_SAD_x0', 'LS_SAD_x0']
    
    for metric in metrics:
        plt.figure()
        for model in model_order:
            model_data = data[data['model'] == model]
            if timesteps:
                model_data = model_data[model_data['inference_timestamp'].isin(timesteps)]
            model_data.sort_values('inference_timestamp')
            mapped_model_name = map_model_names(model)
            color = get_model_color(mapped_model_name)
            metric_values = [float(m) for m in model_data[metric]]
            if timesteps:
                timesteps_values = [str(t) for t in model_data['inference_timestamp']]
                print(timesteps_values)
                print(metric_values)
                plt.plot(timesteps_values, metric_values, label=mapped_model_name, color=color, marker='o')
            
            else:
                plt.plot(model_data['inference_timestamp'], metric_values, label=mapped_model_name, color=color, marker='o')

        plt.title(f'{metric.capitalize()} over Timesteps for {dataset} dataset')
        plt.xlabel('Inference Timestep')
        plt.ylabel(metric.capitalize())
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Legend outside the plot
        plt.grid(False)  # Remove background grid
       
        y_min, y_max = plt.gca().get_ylim()
        y_ticks = np.linspace(y_min, y_max, num=6)
        plt.yticks(y_ticks)

        if timesteps is not None:
            plt.xticks(range(len(timesteps)), [str(t) for t in timesteps])

        # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))  # Format y-axis
        plt.savefig(os.path.join(output_dir, f'{dataset}-{metric}.png'), bbox_inches='tight')
        plt.close()

def main(timesteps=None, output_dir='graphs'):
    # Create the output directory if it does not exist
    output_dir = os.path.join(os.getcwd(), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    data = fetch_data()
    for dataset in data['dataset'].unique():
        dataset_data = data[data['dataset'] == dataset]
        plot_and_save_graphs(dataset_data, dataset, timesteps, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and save graphs from inference timestep analysis.')
    parser.add_argument('--timesteps', type=int, nargs='*', default=None, help='List of timesteps to consider (e.g., 1 5 50 100). If not provided, all timesteps will be used.')
    parser.add_argument('--output_dir', type=str, default='graphs', help='Directory to save the generated graphs.')
    
    args = parser.parse_args()
    
    main(timesteps=args.timesteps, output_dir=args.output_dir)
