import pandas as pd
from glob import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Aggregate data")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory of output files to aggregate."
    )
    args = parser.parse_args()
    
    dataframes = []
    
    for filename in glob(args.output_dir + '/*.csv'):
        df = pd.read_csv(filename)
        dataframes.append(df)
    
    df = pd.concat(dataframes).reset_index(drop=True)
    
    df.to_csv(args.output_dir + '/all_results.csv', mode='w', index=False)

if __name__ == "__main__":
    main()