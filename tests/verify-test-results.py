"""Reject empty or failing test runs, including tests removed by a shrinker."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

for directory in sys.argv[1:]:
    reports = list(Path(directory).glob("TEST-*.xml"))
    assert reports, f"No test reports in {directory}"
    for report in reports:
        root = ET.parse(report).getroot()
        tests = int(root.attrib["tests"])
        assert tests > 0, f"No tests executed: {report}"
        assert int(root.attrib.get("failures", 0)) == 0, report
        assert int(root.attrib.get("errors", 0)) == 0, report
        assert int(root.attrib.get("skipped", 0)) < tests, report
        print(f"Verified {tests} executed tests: {report}")
