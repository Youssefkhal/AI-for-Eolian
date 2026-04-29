import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from train import main


if __name__ == "__main__":
    main()
