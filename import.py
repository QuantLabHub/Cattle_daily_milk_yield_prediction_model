#Acessing the columns
farm_csv = pd.read_csv('janus_farm_3_month_cattle_dataset.csv')
farm_age = farm_csv['age_years']
farm_body_weight=farm_csv['body_weight_kg']
farm_milk_yeild=farm_csv['daily_milk_yield_l']
farm_kcal=farm_csv['daily_energy_intake_kcal']

#mean value
maen_age = farm_age.mean()
mean_body_weight = farm_body_weight.mean()
mean_milk_yeild = farm_milk_yeild.mean()
mean_kcal = farm_kcal.mean()
#print_values
print(f"Mean Age: {maen_age}")
print(f"Mean Body Weight: {mean_body_weight}")
print(f"Mean Milk Yeild: {mean_milk_yeild}")
print(f"Mean Kcal: {mean_kcal}")
#VALUES
'''
Mean Age: 6.3950000000000005
Mean Body Weight: 485.78333333333336
Mean Milk Yeild: 16.672753623188406
Mean Kcal: 32190.99782608695
'''
#standard_deviation
std_age = farm_age.std()
std_body_weight = farm_body_weight.std()
std_milk_yeild = farm_milk_yeild.std()
std_kcal = farm_kcal.std()
#print_values
print(f"Standard Deviation Age: {std_age}")
print(f"Standard Deviation Body Weight: {std_body_weight}")
print(f"Standard Deviation Milk Yeild: {std_milk_yeild}")
print(f"Standard Deviation Kcal: {std_kcal}")
#VALUES
'''
Standard Deviation Age: 1.8066727197701384
Standard Deviation Body Weight: 109.58195010185622
Standard Deviation Milk Yeild: 7.846738744395597
Standard Deviation Kcal: 7342.49559045959
'''
