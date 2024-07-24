import streamlit as st
from crewai import Crew
from textwrap import dedent
from agents import TravelAgents
from tasks import TravelTasks

from dotenv import load_dotenv
load_dotenv()
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

linkedin_link = "[LinkedIn](https://www.linkedin.com/in/muhammad-talha-806126234/)"

# st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown(hide_st_style, unsafe_allow_html=True)

# Your Streamlit app content here
st.sidebar.markdown(linkedin_link, unsafe_allow_html=True)

class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.origin = origin
        self.cities = cities
        self.date_range = date_range
        self.interests = interests

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = TravelAgents()
        tasks = TravelTasks()

        # Define your custom agents and tasks here
        expert_travel_agent = agents.expert_travel_agent()
        city_selection_expert = agents.city_selection_expert()
        local_tour_guide = agents.local_tour_guide()

        # Custom tasks include agent name and variables as input
        plan_itinerary = tasks.plan_itinerary(
            expert_travel_agent,
            self.cities,
            self.date_range,
            self.interests
        )

        identify_city = tasks.identify_city(
            city_selection_expert,
            self.origin,
            self.cities,
            self.interests,
            self.date_range
        )

        gather_city_info = tasks.gather_city_info(
            local_tour_guide,
            self.cities,
            self.date_range,
            self.interests
        )

        # Define your custom crew here
        crew = Crew(
            agents=[expert_travel_agent,
                    city_selection_expert,
                    local_tour_guide
                    ],
            tasks=[
                plan_itinerary,
                identify_city,
                gather_city_info
            ],
            verbose=True,
        )

        result = crew.kickoff()
        return result

st.title("Welcome to Trip Planner Crew")
st.write('-------------------------------')

origin = st.text_input("From where will you be traveling from?")
cities = st.text_input("What are the cities options you are interested in visiting?")
date_range = st.text_input("What is the date range you are interested in traveling?")
interests = st.text_input("What are some of your high level interests and hobbies?")

if st.button("Plan Trip"):
    trip_crew = TripCrew(origin, cities, date_range, interests)
    result = trip_crew.run()
    
    st.write("## Here is your Trip Plan")
    st.info(result)
