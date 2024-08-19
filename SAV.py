import pandas as pd
import pyreadstat

# Load the .sav file
file_path = 'path_to_your_file.sav'
df, meta = pyreadstat.read_sav(file_path)


print(df.head())
