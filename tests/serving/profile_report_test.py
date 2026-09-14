"""nsys stop returning zero does not guarantee that its importer succeeded."""

import pytest
from scripts.local_batch_sweep import validate_profile_report


@pytest.mark.parametrize("partial_report", [False, True])
def test_reject_missing_or_partial_report_after_importer_error(tmp_path, partial_report):
    (tmp_path / "server.stdout.log").write_text(
        "Importer error status: An unknown error occurred.\n"
    )
    if partial_report:
        (tmp_path / "timeline.nsys-rep").write_bytes(b"incomplete metadata")
    with pytest.raises(RuntimeError, match="Nsight report import failed"):
        validate_profile_report(tmp_path)
