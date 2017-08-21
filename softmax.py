import math

#If we take an input of [1, 2, 3, 4, 1, 2, 3], the softmax of that is [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
# scores
z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 0.1,]
z_exp = [math.exp(i) for i in z]

print([round(i, 2) for i in z_exp])
#[2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]

sum_z_exp = sum(z_exp)
print(round(sum_z_exp, 2))
#114.98

softmax = [round(i / sum_z_exp, 3) for i in z_exp]
print(softmax)
#[0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]