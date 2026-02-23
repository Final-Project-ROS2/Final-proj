#!/usr/bin/env bash

set -e

SESSION_NAME="final-project"
WORKSPACE_DIR="$HOME/final_project_ws"
TARGET_SUBDIR="$1"
TARGET_DIR="$WORKSPACE_DIR/src"

if [[ -n "$TARGET_SUBDIR" && -d "$WORKSPACE_DIR/src/$TARGET_SUBDIR" ]]; then
	TARGET_DIR="$WORKSPACE_DIR/src/$TARGET_SUBDIR"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
	tmux attach -t "$SESSION_NAME"
	exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -c "$WORKSPACE_DIR"

# Create a 2x2 grid.
tmux split-window -h -c "$WORKSPACE_DIR"
tmux select-pane -t 0
tmux split-window -v -c "$WORKSPACE_DIR"
tmux select-pane -t 1
tmux split-window -v -c "$WORKSPACE_DIR"

# Top left: run project.
tmux send-keys -t 0 "cd $WORKSPACE_DIR" C-m
tmux send-keys -t 0 "source ./vision_venv/bin/activate" C-m

# Top right: call ros2 action.
tmux send-keys -t 1 "cd $WORKSPACE_DIR" C-m

# Bottom left: colcon build.
tmux send-keys -t 2 "cd $WORKSPACE_DIR" C-m
tmux send-keys -t 2 "source ./vision_venv/bin/activate" C-m

# Bottom right: access code.
tmux send-keys -t 3 "cd $WORKSPACE_DIR" C-m
tmux send-keys -t 3 "source ./vision_venv/bin/activate" C-m
tmux send-keys -t 3 "cd $TARGET_DIR" C-m

tmux select-pane -t 2
tmux attach -t "$SESSION_NAME"
