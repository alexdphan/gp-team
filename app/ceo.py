# ceo.py
"""Module containing the CEO class."""

from .models import GPTeamMember

class CEO(GPTeamMember):
    """CEO class inherits from GPTeamMember class."""

    def __init__(self, role="CEO"):
        super().__init__(role)

    def assign_task(self, task, team_members):
        """Assigns a task to a team member."""
        # Implement your task assignment logic here
        # e.g., return appropriate GPTeamMember based on the task content
        # For simplicity, we return the first member

        return team_members[0]