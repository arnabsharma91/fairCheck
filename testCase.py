from FairnessTestCases import DecisionTreeCredit, DecisionTreeAdult
from fairCheck import Assume, Assert, propCheck

propCheck(no_of_params=2, max_samples=1500, model_type='sklearn', model=DecisionTreeAdult.func_main(),
          mul_cex=True, xml_file='dataInput.xml')

for i in range(0, 13):
    if i == 8:
        Assume('x[i] != y[i]', i)
    else:
        Assume('x[i] = y[i]', i)
Assert('model.predict(x) == model.predict(y)')