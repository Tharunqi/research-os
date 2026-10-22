
from app.skills.paper_comparison import compare_papers

result = compare_papers(
    paper_ids=[2, 3],  # Replace with two actual paper IDs
    project_id=3,
)

print("Grounded:", result["grounded"])
print("\nComparison:\n", result["comparison"])
print("\nSources:\n", result["sources"])
print("\nError:\n", result.get("error"))
