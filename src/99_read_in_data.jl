using DataFrames
using Arrow
using Dates

LocalResults = DataFrame(Arrow.Table("./Data/Processed/LR.arr"));
NationalResults =DataFrame(Arrow.Table("./Data/Processed/NR.arr"));
EuropeanParliament = DataFrame(Arrow.Table("./Data/Processed/EPE.arr"));
Opinions = DataFrame(Arrow.Table("./Data/Processed/Opinions.arr"));