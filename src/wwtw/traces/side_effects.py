"""Which AgentDojo v1.2.2 tools have security-relevant side effects (monitor-gated)."""

SIDE_EFFECTING: dict[str, frozenset[str]] = {
    "workspace": frozenset(
        {
            "send_email",
            "delete_email",
            "create_calendar_event",
            "cancel_calendar_event",
            "reschedule_calendar_event",
            "add_calendar_event_participants",
            "create_file",
            "delete_file",
            "append_to_file",
            "share_file",
        }
    ),
    "travel": frozenset(
        {
            "reserve_hotel",
            "reserve_car_rental",
            "reserve_restaurant",
            "create_calendar_event",
            "cancel_calendar_event",
            "send_email",
        }
    ),
    "banking": frozenset(
        {
            "send_money",
            "schedule_transaction",
            "update_scheduled_transaction",
            "update_password",
            "update_user_info",
        }
    ),
    "slack": frozenset(
        {
            "send_direct_message",
            "send_channel_message",
            "add_user_to_channel",
            "invite_user_to_slack",
            "remove_user_from_slack",
            "post_webpage",
            "get_webpage",  # logs the request URL: an exfiltration channel
        }
    ),
}

READ_ONLY: dict[str, frozenset[str]] = {
    "workspace": frozenset(
        {
            "get_current_day",
            "get_day_calendar_events",
            "get_draft_emails",
            "get_file_by_id",
            "get_received_emails",
            "get_sent_emails",
            "get_unread_emails",
            "list_files",
            "search_calendar_events",
            "search_contacts_by_email",
            "search_contacts_by_name",
            "search_emails",
            "search_files",
            "search_files_by_filename",
        }
    ),
    "travel": frozenset(
        {
            "check_restaurant_opening_hours",
            "get_all_car_rental_companies_in_city",
            "get_all_hotels_in_city",
            "get_all_restaurants_in_city",
            "get_car_fuel_options",
            "get_car_price_per_day",
            "get_car_rental_address",
            "get_car_types_available",
            "get_contact_information_for_restaurants",
            "get_cuisine_type_for_restaurants",
            "get_day_calendar_events",
            "get_dietary_restrictions_for_all_restaurants",
            "get_flight_information",
            "get_hotels_address",
            "get_hotels_prices",
            "get_price_for_restaurants",
            "get_rating_reviews_for_car_rental",
            "get_rating_reviews_for_hotels",
            "get_rating_reviews_for_restaurants",
            "get_restaurants_address",
            "get_user_information",
            "search_calendar_events",
        }
    ),
    "banking": frozenset(
        {
            "get_balance",
            "get_iban",
            "get_most_recent_transactions",
            "get_scheduled_transactions",
            "get_user_info",
            "read_file",
        }
    ),
    "slack": frozenset(
        {"get_channels", "get_users_in_channel", "read_channel_messages", "read_inbox"}
    ),
}

# Read tools that mutate state only as bookkeeping (not security-relevant).
BOOKKEEPING: dict[str, frozenset[str]] = {"workspace": frozenset({"get_unread_emails"})}


def is_side_effecting(suite: str, function: str) -> bool:
    return function in SIDE_EFFECTING.get(suite, frozenset())
