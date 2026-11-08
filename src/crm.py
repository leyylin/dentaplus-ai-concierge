import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class BaseCRMClient(ABC):
    @abstractmethod
    async def create_lead(
        self,
        name: str,
        phone: str,
        service: str,
        slot_datetime: str,
    ) -> Optional[int]:
        """Creates a lead in CRM and returns created lead ID or None on error."""
        pass

    @abstractmethod
    async def update_lead_status(self, lead_id: int, status_id: int) -> bool:
        """Updates lead status in CRM. Returns True if successful, False otherwise."""
        pass


class AmoCRMClient(BaseCRMClient):
    def __init__(self, subdomain: str, access_token: str) -> None:
        self.subdomain = subdomain
        self.access_token = access_token
        self.base_url = f"https://{subdomain}.amocrm.ru"

    async def create_lead(
        self,
        name: str,
        phone: str,
        service: str,
        slot_datetime: str,
    ) -> Optional[int]:
        endpoint = f"{self.base_url}/api/v4/leads/complex"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        lead_custom_fields: list[dict[str, Any]] = [
            {
                "field_id": settings.AMO_CRM_FIELD_SERVICE_ID,
                "values": [{"value": service}],
            }
        ]

        try:
            dt = datetime.fromisoformat(slot_datetime)
            lead_custom_fields.append(
                {
                    "field_id": settings.AMO_CRM_FIELD_VISIT_TIME_ID,
                    "values": [{"value": int(dt.timestamp())}],
                }
            )
        except Exception:
            logger.warning("Failed to parse slot_datetime '%s' for amoCRM custom field", slot_datetime)

        lead_data: dict[str, Any] = {
            "name": f"Заявка: {service}",
            "custom_fields_values": lead_custom_fields,
            "_embedded": {
                "contacts": [
                    {
                        "name": name,
                        "custom_fields_values": [
                            {
                                "field_code": "PHONE",
                                "values": [
                                    {
                                        "value": phone,
                                        "enum_code": "WORK",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        }

        payload: list[dict[str, Any]] = [lead_data]

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    lead_id = data[0].get("id")
                    if lead_id is not None:
                        logger.info("amoCRM lead created successfully with ID: %s", lead_id)
                        lead_id_int = int(lead_id)
                        notes_endpoint = f"https://{self.subdomain}.amocrm.ru/api/v4/leads/{lead_id_int}/notes"
                        notes_payload = [
                            {
                                "note_type": "common",
                                "params": {
                                    "text": f"📅 Новая запись на прием:\nУслуга: {service}\nВремя: {slot_datetime}\nПациент: {name}\nТелефон: {phone}"
                                },
                            }
                        ]
                        try:
                            note_response = await client.post(
                                notes_endpoint, json=notes_payload, headers=headers
                            )
                            note_response.raise_for_status()
                        except Exception as exc:
                            logger.warning(
                                "Failed to send note for lead %s in amoCRM: %s",
                                lead_id_int,
                                exc,
                            )
                        return lead_id_int
                logger.error("Unexpected amoCRM response format: %s", data)
                return None
        except httpx.HTTPError as exc:
            logger.error("HTTP error while creating lead in amoCRM: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected error while creating lead in amoCRM: %s", exc)
            return None

    async def update_lead_status(self, lead_id: int, status_id: int) -> bool:
        endpoint = f"https://{self.subdomain}.amocrm.ru/api/v4/leads/{lead_id}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        payload = {"status_id": status_id}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.patch(endpoint, json=payload, headers=headers)
                if response.status_code in (200, 204):
                    logger.info(
                        "amoCRM lead %s status successfully updated to %s",
                        lead_id,
                        status_id,
                    )
                    return True
                logger.error(
                    "Failed to update amoCRM lead %s status to %s: HTTP %s, response: %s",
                    lead_id,
                    status_id,
                    response.status_code,
                    response.text,
                )
                return False
        except httpx.HTTPError as exc:
            logger.error(
                "HTTP error while updating lead %s status in amoCRM: %s",
                lead_id,
                exc,
            )
            return False
        except Exception as exc:
            logger.error(
                "Unexpected error while updating lead %s status in amoCRM: %s",
                lead_id,
                exc,
            )
            return False


class MockCRMClient(BaseCRMClient):
    async def create_lead(
        self,
        name: str,
        phone: str,
        service: str,
        slot_datetime: str,
    ) -> Optional[int]:
        logger.info(
            "MockCRM: Lead created for '%s' (%s), service: '%s', slot: '%s' -> ID 99001",
            name,
            phone,
            service,
            slot_datetime,
        )
        return 99001

    async def update_lead_status(self, lead_id: int, status_id: int) -> bool:
        logger.info(
            "MockCRM: Lead %s status updated to %s",
            lead_id,
            status_id,
        )
        return True


def get_crm_client() -> BaseCRMClient:
    if (
        settings.AMO_CRM_ENABLED
        and settings.AMO_CRM_SUBDOMAIN.strip()
        and settings.AMO_CRM_ACCESS_TOKEN.strip()
    ):
        return AmoCRMClient(
            subdomain=settings.AMO_CRM_SUBDOMAIN.strip(),
            access_token=settings.AMO_CRM_ACCESS_TOKEN.strip(),
        )
    return MockCRMClient()
