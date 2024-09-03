# Copyright Sierra

import json
from typing import Any, Dict


def cancel_reservation(
    data: Dict[str, Any],
    reservation_id: str,
) -> str:
    reservations = data["reservations"]
    if reservation_id not in reservations:
        return "Error: reservation not found"
    reservation = reservations[reservation_id]

    # reverse the payment
    refunds = []
    for payment in reservation["payment_history"]:
        refunds.append(
            {
                "payment_id": payment["payment_id"],
                "amount": -payment["amount"],
            }
        )
    reservation["payment_history"].extend(refunds)
    reservation["status"] = "cancelled"
    return json.dumps(reservation)


cancel_reservation.__info__ = {
    "type": "function",
    "function": {
        "name": "cancel_reservation",
        "description": "Cancel the whole reservation.",
        "parameters": {
            "type": "object",
            "properties": {
                "reservation_id": {
                    "type": "string",
                    "description": "The reservation ID, such as 'ZFA04Y'.",
                },
            },
            "required": ["reservation_id"],
        },
    },
}
