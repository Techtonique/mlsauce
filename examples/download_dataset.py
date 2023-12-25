import mlsauce as ms 

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

print("\n====================================================== \n")

# Affairs dataset
df3 = ms.download(pkgname="AER", dataset="Affairs", source="https://zeileis.r-universe.dev/")
print(f"===== df3: \n {df3} \n")
print(f"===== df3.dtypes: \n {df3.dtypes}")
