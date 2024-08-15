import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")




def get_month_value(month):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return months.index(month)





def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    with open(filename, "r") as data_file:
        reader = csv.DictReader(data_file)
        for row in reader:
            #the first 17 columns are evidence. Read them as user_evidence
            user_evidence = []
            user_evidence.append(int(row["Administrative"]))
            user_evidence.append(float(row["Administrative_Duration"]))
            user_evidence.append(int(row["Informational"]))
            user_evidence.append(float(row["Informational_Duration"]))
            user_evidence.append(int(row["ProductRelated"]))
            user_evidence.append(float(row["ProductRelated_Duration"]))
            user_evidence.append(float(row["BounceRates"]))
            user_evidence.append(float(row["ExitRates"]))
            user_evidence.append(float(row["PageValues"]))
            user_evidence.append(float(row["SpecialDay"]))
            month = get_month_value(row["Month"])
            user_evidence.append(month)
            user_evidence.append(int(row["OperatingSystems"]))
            user_evidence.append(int(row["Browser"]))
            user_evidence.append(int(row["Region"]))
            user_evidence.append(int(row["TrafficType"]))
            visitor_type = 1 if row["VisitorType"] == "Returning_Visitor" else 0
            user_evidence.append(visitor_type)
            weekend = 1 if row["Weekend"] == "TRUE" else 0
            user_evidence.append(weekend)

            # add the evidence for the user to the collective evidence list
            evidence.append(user_evidence)

            # the last column is the label
            revenue = 1 if row["Revenue"] == "TRUE" else 0
            labels.append(revenue)

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors= 1)
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    size = len(labels)
    total_positive = 0

    total_true = [0, 0]  # total_true[0] = times true negative
                         # total_true[1] = times true positive
    for i in range(size):
        true_value = labels[i]       # 1 - user purchased,    0 - user did not purchase
        total_positive += true_value
        if true_value == predictions[i]:
            total_true[labels[i]] += 1


    if total_positive != 0:
        sensitivity = total_true[1] / total_positive
    else:
        sensitivity = 1

    if total_positive != size:
        specificity = total_true[0] / (size - total_positive)
    else:
        specificity = 1

    return sensitivity, specificity


if __name__ == "__main__":
    main()
