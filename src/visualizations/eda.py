import matplotlib.pyplot as plt
import seaborn as sns

def mass_histplot_features(df, hue_col, bin_cnt=30):
    for col in df.columns:
        plt.figure(figsize=(10, 6))
        # df.assign and onwards is derived from https://stackoverflow.com/questions/45201514/how-to-edit-a-seaborn-legend-title-and-labels-for-figure-level-functions
        ax = sns.histplot(data=df.assign(high_quality=df[hue_col].map({0: "No", 1: 'Yes'})), x=col, hue='high_quality', bins=bin_cnt, kde=True, palette="viridis")
        plt.title(f'Distribution of {col} by Good or Bad Quality')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()