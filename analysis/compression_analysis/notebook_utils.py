import pandas as pd 
import matplotlib.pyplot as plt 


def json_files_to_single_df(json_list):
    return pd.concat([pd.read_json(f) for f in json_list], ignore_index=True)


def quality_vs_compression(data, metric, time_scale="ms"):
    filtered_data = data[data.metric_used == metric] 
    all_means = {} 
    all_mins = {} 
    for comp in filtered_data.compression_method.unique():
        all_means[comp] = filtered_data[filtered_data.compression_method == comp].mean(numeric_only=True)
        all_mins[comp] = filtered_data[filtered_data.compression_method == comp].min(numeric_only=True)
    all_means = pd.DataFrame(all_means).transpose().reset_index() 
    all_mins = pd.DataFrame(all_mins).transpose().reset_index()
    all_means['min_quality'] = all_mins['quality_value']
    all_means['min_compression'] = all_mins['compression_rate']
    if time_scale == "ms":
        all_means['compression_time'] = 1000 * all_means['compression_time']
    elif time_scale == "s":
        all_means['compression_time'] = all_means['compression_time']
    ax = all_means.plot(
        kind="bar", 
        x = "index",
        y = ["min_compression", "compression_rate", "min_quality", "quality_value", "compression_time"],
        color=["green", "blue", "orange", "red", "pink"],
        figsize=(12,8),
        title=metric + " with time in " + time_scale
    )
    # to show values on bars 
    for container in ax.containers:
        ax.bar_label(container)
    # to show rotated indices 
    plt.xticks(rotation=30)
    return ax 