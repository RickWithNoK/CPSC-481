from diagnostics import Diagnostics

d = Diagnostics()

result = d.diagnose(
    "No",
    "Yes",
    "Abnormal",
    "Absent"
)

print(result)