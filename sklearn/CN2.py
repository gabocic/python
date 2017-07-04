import Orange
# Read some data
titanic = Orange.data.Table("titanic")

# construct the learning algorithm and use it to induce a classifier
cn2_learner = Orange.classification.rules.CN2Learner()
cn2_clasifier = cn2_learner(titanic)

# All rule-base classifiers can have their rules printed out like this:
for r in cn2_classifier.rules:
        print Orange.classification.rules.rule_to_string(r)
