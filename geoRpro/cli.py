from typing import Generator, Any, Final, Literal
import argparse
import traceback
import os
import sys
import json
from collections import OrderedDict
from geoRpro.workflow import Workflow
import pdb


def cli():
    parser = argparse.ArgumentParser(
        description="Execute the workflow defined in the json fie"
    )
    parser.add_argument("--path", required=True)

    return parser


def main() -> None:
    parser = cli()
    args = parser.parse_args()
    sourcefile = args.path

    # check whether sourcedir exist
    if not os.path.exists(sourcefile):
        print("Source file was not found")
        sys.exit()

    try:
        with open(sourcefile) as json_file:
            wf_data: dict[str, Any] = json.load(
                json_file, object_pairs_hook=OrderedDict
            )
    except Exception as e:
        traceback.print_exc()
        print("Something wrong with the json file")
        sys.exit()

    wf = Workflow(wf_data)

    try:
        wf.run_workflow()
    except Exception as e:
        traceback.print_exc()
        print("Something wrong with the workflow")
        sys.exit()

    print("done")
