import streamlit as st
import sqlite3
import pandas as pd
import altair as alt

# Path to the database
DB_PATH = r"D:\Desktop\Github for resume\LatticeBuild_Project\src\logs\rag_logs.db"

# Load and cache data
@st.cache_data
def load_rag_logs(db_path):
    conn = sqlite3.connect(db_path)
    query = """
        SELECT 
            id, timestamp, query_exec_time, model_name, vector_db, local_use
        FROM rag_logs;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def main():
    st.set_page_config(page_title="RAG Log Viewer", layout="wide")
    st.title("📊 RAG Logs Dashboard")

    # Load data
    df = load_rag_logs(DB_PATH)

    if df.empty:
        st.warning("No data found in the `rag_logs` table.")
        return

    # Section: Table View
    st.markdown("### 🗂️ Log Records")
    with st.expander("Click to view full log table"):
        st.dataframe(df, use_container_width=True, height=400)

    st.markdown("---")

    # Section: Bar Chart (Avg Query Time)
    st.markdown("### 📈 Average Query Time per Model")
    st.markdown("&nbsp;", unsafe_allow_html=True)  # spacing

    avg_runtime_df = df.groupby("model_name", as_index=False)["query_exec_time"].mean()
    avg_runtime_df.rename(columns={"query_exec_time": "avg_query_time"}, inplace=True)

    chart = alt.Chart(avg_runtime_df).mark_bar(size=40).encode(
        x=alt.X("model_name:N", title="Model Name", sort="-y", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("avg_query_time:Q", title="Avg Query Time (ms)"),
        tooltip=["model_name", "avg_query_time"]
    ).properties(
        width="container",
        height=400
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )

    st.altair_chart(chart, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # Section: Scatter Plot with Model Filter
    st.markdown("### 🎯 Query Runtime Scatter Plot by Model")

    available_models = sorted(df["model_name"].unique())
    selected_models = st.multiselect(
        "Select Models to Display",
        options=available_models,
        default=available_models
    )

    if selected_models:
        filtered_df = df[df["model_name"].isin(selected_models)]

        scatter_chart = alt.Chart(filtered_df).mark_circle(size=60).encode(
            x=alt.X("timestamp:T", title="Timestamp"),
            y=alt.Y("query_exec_time:Q", title="Query Execution Time (ms)"),
            color=alt.Color("model_name:N", title="Model Name"),
            tooltip=["model_name", "query_exec_time", "timestamp"]
        ).properties(
            width="container",
            height=400
        ).interactive()

        st.altair_chart(scatter_chart, use_container_width=True)
    else:
        st.info("Please select at least one model to view the scatter plot.")

    st.markdown("<br><br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
