"""Protect attribution against shared HLO metadata and overlapping host ranges."""

from scripts.analyze_static_stages import HloStages, correlate_apis, stage_name

HLO = """HloModule demo
%shared (p: f32[]) -> f32[] {
  %p = f32[] parameter(0)
  ROOT %reuse = f32[] copy(%p), metadata={op_name="armory_stage_vlm_prefill/copy"}
}
%step (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %fusion = f32[] fusion(%x), calls=%shared, metadata={op_name="armory_stage_action/while/body/add"}
}
ENTRY %main (a: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  ROOT %loop = f32[] while(%a), body=%step, metadata={op_name="armory_stage_action/while"}
}
"""


def test_shared_fusion_uses_executing_loop_context():
    resolver = HloStages(HLO)
    assert resolver.label_tags("Thunk:#hlo_op=fusion#") == {"vlm_prefill", "action"}
    assert resolver.action_loop_names() == ["loop"]
    api = dict(start=30, end=35, tid=7, pid=1, correlation=2)
    ranges = [
        (10, 80, "Thunk:#hlo_op=loop_body#", 7),
        (20, 50, "Thunk:#hlo_op=fusion#", 7),
        (25, 40, "Thunk:#name=armory_stage_vlm_embed,hlo_op=else#", 8),
    ]
    result = correlate_apis([api], ranges, resolver)[(1, 2)]
    assert result["stage"] == "action"
    assert result["label"] == "Thunk:#hlo_op=loop_body#"


def test_overlapping_apis_do_not_remove_ranges_needed_by_inner_api():
    resolver = HloStages(HLO)
    apis = [
        dict(start=15, end=90, tid=7, pid=1, correlation=1),
        dict(start=20, end=30, tid=7, pid=1, correlation=2),
    ]
    ranges = [
        (10, 100, "Thunk:#name=armory_stage_vlm_embed#", 7),
        (12, 35, "Thunk:#hlo_op=loop_body#", 7),
    ]
    result = correlate_apis(apis, ranges, resolver)
    assert result[(1, 1)]["stage"] == "vlm_embed"
    assert result[(1, 2)]["stage"] == "action"
    assert stage_name({"vlm_embed", "vlm_prefill"}) == "vlm_mixed"
    assert stage_name(set()) == "unattributed"
