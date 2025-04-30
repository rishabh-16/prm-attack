VERBOSE_TASK = """Task: Given a mathematical question and a set of solution steps, rewrite both the question and each step so they are more detailed and descriptive, without adding any new theorems, equations, or assumptions."""
CONSISE_TASK = """Task: Given an input mathematical question and its associated solution steps, revise each solution step to be more concise while preserving all original information and without introducing any new theorems, equations, or unstated assumptions."""
EQ_TO_TEXT_TASK = """Task: Given an input mathematical question and its associated solution steps, convert every mathematical equation, numeric value, and symbol within each step into fully written-out text descriptions, spelling out all elements clearly as if explaining them in words."""
CHANGE_NUMBERS_TASK = """Task: Given an input mathematical question and its associated solution steps, replace every numeric value in each solution step with a randomly selected number so that the resulting equations become mathematically incorrect, while preserving the original structure of operations and variables."""

POST_AUGMENTATION_PROMPT = """Requirements:
1. Preserve the original logical structure and the ordering of the question and solution steps.
2. Do not add any new mathematical statements, theorems, equations, or hidden assumptions beyond what is provided.
3. If a portion cannot be modified without changing its original meaning, leave it exactly as is.
4. Wrap your entire response in <response> tags as shown in the example below.
5. Label the question and each step with numbered tags: <question> for the expanded question. <step1>, <step2>, <step3>, etc., for each corresponding step.
6. Never correct incorrectly written steps unless explicitly stated.

Output Format:
<response>
  <question>Your transformed question text here.</question>
  <step1>Your transformed first step here.</step1>
  <step2>Your transformed second step here.</step2>
  <step3>Your transformed third step here.</step3>
  <!-- Continue for all provided steps -->
</response>"""

EQUIVALENCE_PROMPT = """You are given two versions of a mathematical problem and its associated solution steps, which we’ll call Set A and Set B. Your job is to determine whether Set B faithfully reproduces the same meaning, logical flow, and conclusions as Set A, without adding, removing, or altering any essential content.

Instructions:
1. Compare Questions
- Check that the wording of the question in Set B asks for the same quantity or result as Set A.
2. Compare Steps
- For each step in Set A, locate the corresponding step in Set B.
- Verify that each operation, assumption, and conclusion in Set A is mirrored in Set B (even if phrased differently).
- Ensure no extra steps have been inserted and none have been omitted.
3. Assess Logical Equivalence
- Decide whether every inference in Set A has an equivalent inference in Set B.
- Flag any place where Set B’s reasoning diverges in structure or result.
4. Report
- Y: if questions match and every step in Set B corresponds exactly (in meaning and order) to Set A.
- N: if you find any mismatch in the question’s intent or in the logical sequence of steps.

Output Format:
<response> 
    <question> Y or N </question>
    <step1> Y or N </step1>
    <step2> Y or N </step2>
    <step3> Y or N </step3>
    <!-- Continue for all provided steps -->
</response>
"""