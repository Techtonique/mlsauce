import mlsauce as ms 
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")


# `ms.download` parameters 
# pkgname="MASS"
# dataset="Boston"
# source="https://cran.r-universe.dev/"

# the controversial Boston dataset 
df1 = ms.download(dataset="Boston")

print(f"===== df1: \n {df1} \n")
print(f"===== df1.dtypes: \n {df1.dtypes}")

print("\n====================================================== \n")

# Insurance dataset 
df2 = ms.download(dataset="Insurance")
print(f"===== df2: \n {df2} \n")
print(f"===== df2.dtypes: \n {df2.dtypes}")

