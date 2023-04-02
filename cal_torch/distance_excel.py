import pandas as pd
import openpyxl


def save_distance_to_excel(distances, class_names, file_name):
    # Create pandas DataFrame from distance matrix
    df = pd.DataFrame(distances, columns=class_names, index=class_names)

    # Save DataFrame to Excel file
    writer = pd.ExcelWriter(file_name, engine='openpyxl')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

    print(f"Distance matrix saved to {file_name}")
