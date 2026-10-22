# impurities measure how mixed the labels are within a group
# e.g. Imagine a group of 10 students:
#  Group A: 5 pass, 5 fail → highly mixed.
#  Group B: 9 pass, 1 fail → less mixed.
#  Group C: 10 pass, 0 fail → completely pure.
#  Decision Tree prefers splits that make the resulting groups purer, meaning each group contains more consistent labels.
