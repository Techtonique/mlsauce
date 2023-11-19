import mlsauce as ms 

# `ms.download` parameters 
# pkgname="MASS"
# dataset="Boston"
# source="https://cran.r-universe.dev/"

# the controversial Boston data set 
df1 = ms.download(dataset="Boston")

print(f"===== df1: \n {df1} \n")
print(f"===== df1.dtypes: \n {df1.dtypes}")

print("\n====================================================== \n")

# the controversial Boston data set 
df2 = ms.download(dataset="Insurance")
print(f"===== df2: \n {df2} \n")
print(f"===== df2.dtypes: \n {df2.dtypes}")
