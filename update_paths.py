import csv

def add_root_path(file_path, root_path="sign_data/"):
    """
    Prepends a root path to the first two columns of a CSV file.
    """
    updated_rows = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row: # To handle empty rows
                # Check if the path is already prepended
                if not row[0].startswith(root_path):
                    row[0] = root_path + row[0]
                if not row[1].startswith(root_path):
                    row[1] = root_path + row[1]
                updated_rows.append(row)

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)

if __name__ == "__main__":
    train_file = "sign_data/train.csv"
    test_file = "sign_data/test.csv"

    add_root_path(train_file, root_path="sign_data/train/")
    print(f"Updated {train_file}")

    add_root_path(test_file, root_path="sign_data/test/")
    print(f"Updated {test_file}") 