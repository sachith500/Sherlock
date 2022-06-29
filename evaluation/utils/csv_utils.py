import csv


class CSVUtils:

    @classmethod
    def read_csv(cls, file_name):
        file = open(file_name)
        csv_reader = csv.reader(file)

        first_row = True
        data_rows = []
        for row in csv_reader:
            if first_row:
                first_row = False
                continue

            data_rows.append(row)

        return data_rows
