import pandas as pd
import numpy as np
import streamlit as st
import time
import re
from datetime import datetime
from scipy.optimize import Bounds, LinearConstraint, milp

SHEET_ID = "13R2h50G2D4R4822F2Rbq-c9D4VW5ahgABRrpyCfhHYE"
URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

TOPICS = {
    0: "Evaluating Google HEIR for encrypted data processing",
    1: "Optimizing cryptographic Rust code and proving it correct",
    2: "Comparing and implementing MPC-ORAM constructions",
    3: "Burnt-in Subtitle Removal (via Inpainting)",
    4: "Automatic Video Cropping",
    5: "Effect of image and video conditions on landmark-detection performance",
    6: "Robustness to participant characteristics and appearance",
    7: "Model accuracy versus computational efficiency",
    8: "Eye-tracking with event-based sensing",
    9: "Simulation of event-based data",
    10: "Seizure insight online platform",
    11: "Listen to the silent speech",
    12: "From noisy point clouds to closed surfaces",
    13: "Fast coverage path planning",
}

PREFERENCES_COLUMN = "Automatically generated!"
MIN_GROUP_SIZE = 6
MAX_GROUP_SIZE = 7

def parse_priorities(val):
    if not isinstance(val, str):
        return None
    match = re.search(r"\[\s*(\d+(?:\s*,\s*\d+){13})\s*\]", val)
    if not match:
        return None
    priorities = [int(value.strip()) for value in match.group(1).split(",")]
    if sorted(priorities) != list(range(1, len(TOPICS) + 1)):
        return None
    return priorities


def has_student_name(value):
    return isinstance(value, str) and bool(value.strip())


def load_sheet():
    raw_df = pd.read_csv(URL, header=None)
    header_rows = raw_df.index[
        raw_df.eq(PREFERENCES_COLUMN).any(axis=1)
    ].tolist()
    if not header_rows:
        raise ValueError(f'Could not find the "{PREFERENCES_COLUMN}" column.')

    header_row = header_rows[0]
    headers = []
    for index, value in enumerate(raw_df.loc[header_row]):
        headers.append(str(value) if pd.notna(value) else f"unnamed_{index}")

    # Sheet rows 1-3 are layout, headers, and an example; students start at row 4.
    df = raw_df.loc[header_row + 2:].copy()
    df.columns = headers
    name_columns = [
        column
        for column in df.columns
        if "name" in column.lower() and not column.lower().startswith("unnamed_")
    ]
    if not name_columns:
        raise ValueError("Could not find the student-name column.")
    return df, name_columns[0]


def build_distribution(priority_lists):
    rows = []
    for topic_index in TOPICS:
        counts = {rank: 0 for rank in range(1, len(TOPICS) + 1)}
        for priorities in priority_lists:
            counts[priorities[topic_index]] += 1
        rows.append(counts)
    labels = [f"Project {index + 1}: {title}" for index, title in TOPICS.items()]
    return pd.DataFrame(rows, index=labels)


def predict_groups(students):
    if not students:
        return {}, []

    student_count = len(students)
    topic_count = len(TOPICS)
    assignment_count = student_count * topic_count
    topic_offset = assignment_count
    unassigned_offset = topic_offset + topic_count
    variable_count = unassigned_offset + student_count

    objective = np.zeros(variable_count)
    for student_index, student in enumerate(students):
        for topic in TOPICS:
            variable = student_index * topic_count + topic
            objective[variable] = student["priorities"][topic]

    # One unassigned student must always cost more than every possible rank saving.
    unassigned_penalty = student_count * topic_count + 1
    objective[unassigned_offset:] = unassigned_penalty
    objective[topic_offset:unassigned_offset] = 0.001

    rows = []
    lower_bounds = []
    upper_bounds = []

    for student_index in range(student_count):
        row = np.zeros(variable_count)
        start = student_index * topic_count
        row[start:start + topic_count] = 1
        row[unassigned_offset + student_index] = 1
        rows.append(row)
        lower_bounds.append(1)
        upper_bounds.append(1)

    for topic in TOPICS:
        group_size_row = np.zeros(variable_count)
        for student_index in range(student_count):
            group_size_row[student_index * topic_count + topic] = 1

        minimum_row = group_size_row.copy()
        minimum_row[topic_offset + topic] = -MIN_GROUP_SIZE
        rows.append(minimum_row)
        lower_bounds.append(0)
        upper_bounds.append(np.inf)

        maximum_row = group_size_row.copy()
        maximum_row[topic_offset + topic] = -MAX_GROUP_SIZE
        rows.append(maximum_row)
        lower_bounds.append(-np.inf)
        upper_bounds.append(0)

    result = milp(
        c=objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            np.vstack(rows), np.array(lower_bounds), np.array(upper_bounds)
        ),
        options={"time_limit": 20},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Group optimization failed: {result.message}")

    solution = np.rint(result.x).astype(int)
    groups = {}
    unassigned = []
    for student_index, student in enumerate(students):
        assigned_topic = None
        for topic in TOPICS:
            variable = student_index * topic_count + topic
            if solution[variable] == 1:
                assigned_topic = topic
                break

        if assigned_topic is None:
            unassigned.append(student)
        else:
            groups.setdefault(assigned_topic, []).append(student)

    return groups, unassigned


def render_dashboard():
    st.set_page_config(layout="wide")
    st.title("Project Priority Distribution")

    try:
        source_df, name_column = load_sheet()
    except Exception as error:
        st.error(f"Could not load the project preferences: {error}")
        return

    named_df = source_df[source_df[name_column].apply(has_student_name)].copy()
    named_df["priorities"] = named_df[PREFERENCES_COLUMN].apply(parse_priorities)
    valid_df = named_df.dropna(subset=["priorities"]).copy()
    invalid_count = len(named_df) - len(valid_df)

    total_col, valid_col, incomplete_col = st.columns(3)
    total_col.metric("Students with a name", len(named_df))
    valid_col.metric("Complete rankings", len(valid_df))
    incomplete_col.metric("Incomplete/invalid rankings", invalid_count)

    if valid_df.empty:
        st.warning("No complete 14-project rankings were found.")
        return

    priority_lists = valid_df["priorities"].tolist()
    chart_df = build_distribution(priority_lists)

    st.subheader("First-choice demand")
    st.bar_chart(chart_df[1])

    with st.expander("Full ranking distribution"):
        st.dataframe(chart_df, use_container_width=True)

    students = []
    for order, (_, row) in enumerate(valid_df.iterrows()):
        raw_name = row[name_column]
        name = str(raw_name).strip() if pd.notna(raw_name) else ""
        students.append(
            {
                "name": name or f"Unnamed student {order + 1}",
                "priorities": row["priorities"],
                "order": order,
            }
        )

    try:
        groups, unassigned = predict_groups(students)
    except Exception as error:
        st.error(f"Could not optimize the predicted groups: {error}")
        return

    st.header("Predicted groups")
    st.caption(
        "Integer Linear Programming prediction. It minimizes total preference "
        "rank while keeping one group of 6-7 students per displayed topic."
    )

    assigned_ranks = [
        member["priorities"][topic]
        for topic, members in groups.items()
        for member in members
    ]
    group_col, average_col, first_choice_col = st.columns(3)
    group_col.metric("Optimized groups", len(groups))
    average_col.metric(
        "Average assigned rank",
        f"{sum(assigned_ranks) / len(assigned_ranks):.2f}" if assigned_ranks else "-",
    )
    first_choice_col.metric(
        "First-choice assignments", assigned_ranks.count(1)
    )

    group_columns = st.columns(2)
    for display_index, topic in enumerate(sorted(groups)):
        members = groups[topic]
        with group_columns[display_index % 2]:
            with st.expander(
                f"Project {topic + 1}: {TOPICS[topic]} ({len(members)} students)",
                expanded=True,
            ):
                for member in sorted(member["name"] for member in members):
                    st.write(f"- {member}")

    if unassigned:
        st.warning(
            f"{len(unassigned)} students could not be assigned while keeping every "
            f"active topic between {MIN_GROUP_SIZE} and {MAX_GROUP_SIZE} students."
        )
        with st.expander("Unassigned students", expanded=True):
            for student in sorted(unassigned, key=lambda item: item["name"]):
                st.write(f"- {student['name']}")

    if invalid_count:
        st.info(
            f"{invalid_count} named students were not included because their ranking "
            "is incomplete or invalid. A valid ranking must contain every value from "
            "1 through 14 exactly once."
        )

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    render_dashboard()
    time.sleep(10)
    st.rerun()
