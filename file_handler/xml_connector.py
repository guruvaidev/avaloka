import random
from typing import Any

from lxml import etree as ET

from .base_connector import BaseConnector, OutputData
from .utils import convert_type


class XMLConnector(BaseConnector):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = path

    def get_columns(self) -> list[str]:
        # Get columns from the first record
        context = ET.iterparse(self.path, events=("end",))
        for _, elem in context:
            if len(elem):  # has children
                columns = [child.tag for child in elem]
                elem.clear()
                return columns
        return []

    def get_row_count(self) -> int:
        # Count direct children of root
        context = ET.iterparse(self.path, events=("start", "end"))
        count, root = 0, None
        for event, elem in context:
            if root is None and event == "start":
                root = elem
            elif event == "end" and elem is not root and elem in root:
                count += 1
                elem.clear()
        return count

    def get_row(self, row: int) -> list[Any]:
        element_count = -1
        for _, elem in ET.iterparse(self.path, events=("start",)):
            if elem.getparent() is not None and len(elem):  # has children
                element_count += 1
                if element_count == row:
                    return [
                        convert_type(child.text) if child.text is not None else None
                        for child in elem
                    ]

        return []

    def sample_rows(
        self, num_rows: int = 10, random_rows: bool = True
    ) -> list[list[Any]]:
        total_rows = self.get_row_count()
        if not total_rows:
            return []

        rows_to_sample = sorted(
            random.sample(range(total_rows - 1), k=min(num_rows, total_rows - 2))
            if random_rows
            else range(num_rows)
        )

        data = []
        idx, target_idx = 0, 0
        for _, elem in ET.iterparse(self.path, events=("start",)):
            if elem.getparent() is not None and len(elem):  # has children
                if idx == rows_to_sample[target_idx]:
                    data.append(
                        [
                            convert_type(child.text) if child.text is not None else None
                            for child in elem
                        ]
                    )
                    target_idx += 1
                    if target_idx >= len(rows_to_sample):
                        break

                idx += 1
        return data

    def load_data(self) -> list[list[Any]]:
        data = []
        for _, elem in ET.iterparse(self.path, events=("start",)):
            if elem.getparent() is not None and len(elem):  # has children
                data.append(
                    [
                        convert_type(child.text) if child.text is not None else None
                        for child in elem
                    ]
                )
                elem.clear()
        return data

    @staticmethod
    def write_data(data: OutputData, output_path: str) -> None:
        # Write data as XML, root element is <data>, each row is <record>
        root = ET.Element("data")
        for row in data["rows"]:
            record = ET.SubElement(root, "record")
            for col, val in zip(data["columns"], row):
                child = ET.SubElement(record, col)
                child.text = str(val) if val is not None else ""
        tree = ET.ElementTree(root)
        tree.write(output_path, xml_declaration=True, encoding="utf-8")

        