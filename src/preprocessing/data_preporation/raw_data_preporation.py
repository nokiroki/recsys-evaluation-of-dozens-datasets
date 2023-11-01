import json
from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd


class BasePrepare(ABC):

    def __init__(self, raw_path: Path, save_path: Path) -> None:
        self.raw_path = raw_path
        self.save_path = save_path
    
    @abstractmethod
    def make_dataset(self) -> None:
        raise NotImplementedError

class AmazonBooks(BasePrepare):

    def make_dataset(self) -> None:
        pd.read_csv(
            self.raw_path, header=None, names=["userId", "itemId", "rating", "timestamp"]
        ).to_parquet(self.save_path)


class AmazonCds(BasePrepare):

    def make_dataset(self) -> None:
        pd.read_csv(
            self.raw_path, header=None, names=["userId", "itemId", "rating", "timestamp"]
        ).to_parquet(self.save_path)


class AmazonFinefoods(BasePrepare):

    def make_dataset(self) -> None:
        with self.raw_path.open(encoding="latin-1") as file:
            data = file.read()
        
        # Split the data into individual reviews based on the blank line separator
        reviews = data.strip().split("\n\n")

        # Initialize empty lists to store extracted information
        productIds = []
        userIds = []
        reviewScores = []
        reviewTimes = []

        # Parse each review to extract the required information
        for review in reviews:
            lines = review.strip().split("\n")
            extracted_data = {}
            for line in lines:
                if ": " in line:
                    prefix, value = line.split(": ", 1)
                    extracted_data[prefix] = value

            if (
                "product/productId" in extracted_data
                and "review/userId" in extracted_data
                and "review/score" in extracted_data
                and "review/time" in extracted_data
            ):
                productIds.append(extracted_data["product/productId"])
                userIds.append(extracted_data["review/userId"])
                reviewScores.append(float(extracted_data["review/score"]))
                reviewTimes.append(int(extracted_data["review/time"]))

        # Create the DataFrame
        pd.DataFrame(
            {
                "productId": productIds,
                "userId": userIds,
                "reviewscore": reviewScores,
                "time": reviewTimes,
            }
        ).to_parquet(self.save_path)


class AmazonTV(BasePrepare):

    def make_dataset(self) -> None:
        pd.read_csv(
            self.raw_path, header=None, names=["userId", "itemId", "rating", "timestamp"]
        ).to_parquet(self.save_path)


class BeerAdvocate(BasePrepare):

    def make_dataset(self) -> None:
        prime = []
        with self.raw_path.open() as file:
            for line in file.readlines():
                line = line.replace("\'", "\"")
                line = '{' + line[line.find('"beer/beerId"'):]
                line = line[:line.find('"beer/name"')] + line[line.find('"beer/beerId"')+1:]
                line = line[:line.find('"review/text"') - 2] + '}'
                if line == '{}\n':
                    continue
                try:
                    val = json.loads(line)
                    prime.append([
                        val['review/profileName'],
                        val['beer/beerId'],
                        val['review/overall'],
                        val['review/time']
                    ])
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}")
                    print(f"Line: {line}")
                    continue
        ratings = pd.DataFrame(prime, columns=['userId', 'itemId', 'rating', 'timestamp'])
        ratings["rating"] = ratings["rating"].astype(float)
        ratings["timestamp"] = ratings["timestamp"].astype(int)

        ratings.to_parquet(self.save_path, index=False)


class Brightkite(BasePrepare):

    def make_dataset(self) -> None:
        return super().make_dataset()
