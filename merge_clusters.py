"""Merge `kmeans` and `hdbscan` cluster labels from `circuit_clusters.csv`
into `FinalMergedWithCircuits.csv` and write `FinalMergedWithCircuits_with_clusters.csv`.
"""
import pandas as pd


def main():
    final_path = 'FinalMergedWithCircuits.csv'
    clusters_path = 'circuit_clusters.csv'

    final = pd.read_csv(final_path)
    clusters = pd.read_csv(clusters_path)

    # Ensure join keys exist and compatible types
    for df in (final, clusters):
        if 'circuit_id' not in df.columns:
            raise KeyError('circuit_id missing in ' + (final_path if df is final else clusters_path))
        if 'circuit_name' not in df.columns:
            raise KeyError('circuit_name missing in ' + (final_path if df is final else clusters_path))

    # Reduce clusters to keys + labels 
    labels = clusters[['circuit_id', 'circuit_name']].copy()
    if 'kmeans' in clusters.columns:
        labels['kmeans'] = clusters['kmeans']
    if 'hdbscan' in clusters.columns:
        labels['hdbscan'] = clusters['hdbscan']

    # Merge onto final 
    merged = final.merge(labels.drop_duplicates(subset=['circuit_id','circuit_name']),
                         on=['circuit_id','circuit_name'], how='left')

    out_path = 'FinalMergedWithCircuits_with_clusters.csv'
    merged.to_csv(out_path, index=False)
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
