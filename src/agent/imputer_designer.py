from typing import Dict, List

class ImputerDesigner:
    def __init__(self, llm=None):
        self.llm = llm

    def propose_policies(self, mechanism_map: Dict[str,str], numeric: List[str], categorical: List[str]):
        p1 = {"per_column": {c:"IterativeRF" for c in numeric+categorical}, "mnar_shift": 0.00}
        p2 = {"per_column": {**{c:"IterativeRF" for c in numeric}, **{c:"KNN" for c in categorical}}, "mnar_shift": 0.00}
        p3 = {"per_column": {**{c:"IterativeRF" for c in numeric}, **{c:"LLM" for c in categorical}}, "mnar_shift": 0.00}
        if any(v=="MNAR" for v in mechanism_map.values()):
            p4 = {"per_column": p1["per_column"], "mnar_shift": 0.05}
            return [p1, p2, p3, p4]
        return [p1, p2, p3]
