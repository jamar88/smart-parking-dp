"""
Melbourne Open Data API client (Opendatasoft v2.1).

Provides methods to:
- Download the 2019 historical parking sensor dataset (bulk CSV export)
- Fetch live parking bay sensor statuses
- Fetch parking bay GPS coordinates
"""

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BASE_URL = "https://data.melbourne.vic.gov.au/api/explore/v2.1"

DATASET_HISTORICAL = "on-street-car-parking-sensor-data-2019"
DATASET_LIVE_SENSORS = "on-street-parking-bay-sensors"
DATASET_PARKING_BAYS = "on-street-parking-bays"


def _make_session(retries: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    """Create a requests Session with retry logic and exponential backoff.

    Args:
        retries: Maximum number of retry attempts.
        backoff_factor: Multiplier for exponential backoff between retries.

    Returns:
        Configured requests.Session.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class MelbourneAPI:
    """Client for the Melbourne Open Data platform (Opendatasoft v2.1).

    Args:
        base_url: API base URL. Defaults to the Melbourne Open Data endpoint.
        timeout: HTTP request timeout in seconds.
        retries: Number of retry attempts on transient failures.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = 60,
        retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = _make_session(retries=retries)
        logger.debug("MelbourneAPI initialised with base_url=%s", self.base_url)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def download_historical_csv(
        self,
        dest_dir: str | Path = "data/raw",
        filename: str = "parking_2019.csv",
        skip_if_exists: bool = True,
    ) -> Path:
        """Download the 2019 on-street parking sensor dataset.

        The dataset is published as a ZIP archive (~717 MB) on S3.  This method
        streams the ZIP to disk then extracts the single CSV inside it.

        Actual download URL (verified 2026-04):
            https://opendatasoft-s3.s3.amazonaws.com/downloads/archive/7pgd-bdf2.zip

        CSV columns (20 total):
            DeviceId, ArrivalTime, DepartureTime, DurationMinutes,
            StreetMarker, SignPlateID, Sign, AreaName, StreetId, StreetName,
            BetweenStreet1ID, BetweenStreet1, BetweenStreet2ID, BetweenStreet2,
            SideOfStreet, SideOfStreetCode, SideName, BayId,
            InViolation, VehiclePresent

        Args:
            dest_dir: Directory where the extracted CSV will be saved.
            filename: Name for the saved CSV file.
            skip_if_exists: If True, skip download when the CSV already exists.

        Returns:
            Path to the extracted CSV file.
        """
        import zipfile

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        if skip_if_exists and dest_path.exists():
            logger.info("Historical CSV already exists at %s — skipping download.", dest_path)
            return dest_path

        zip_url = "https://opendatasoft-s3.s3.amazonaws.com/downloads/archive/7pgd-bdf2.zip"
        zip_path = dest_dir / "parking_2019.zip"

        # --- Step 1: stream-download the ZIP ---
        logger.info("Downloading historical ZIP from %s", zip_url)
        response = self.session.get(zip_url, stream=True, timeout=self.timeout)
        response.raise_for_status()

        chunk_size = 1024 * 1024  # 1 MB
        downloaded_bytes = 0
        log_interval = 50 * 1024 * 1024  # log every 50 MB
        next_log_at = log_interval

        with zip_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes >= next_log_at:
                        logger.info("Downloaded %.0f MB…", downloaded_bytes / 1024 / 1024)
                        next_log_at += log_interval

        logger.info("ZIP download complete: %.1f MB", downloaded_bytes / 1024 / 1024)

        # --- Step 2: extract the single CSV from the ZIP ---
        logger.info("Extracting CSV from ZIP…")
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            logger.info("ZIP contains: %s", names)
            csv_name = next(n for n in names if n.lower().endswith(".csv"))
            with zf.open(csv_name) as src, dest_path.open("wb") as dst:
                while chunk := src.read(chunk_size):
                    dst.write(chunk)

        zip_path.unlink()  # remove ZIP after successful extraction
        logger.info("Extracted to %s", dest_path)
        return dest_path

    def get_live_sensors(self) -> pd.DataFrame:
        """Fetch all current parking bay sensor statuses.

        Paginates the ``on-street-parking-bay-sensors`` dataset using
        limit=100 / offset increments until all records are retrieved.

        Live API schema (verified 2026-04):
            kerbsideid        → bay_id
            status_description → status  ("Unoccupied" / "Present")
            zone_number       → zone_number
            location.lat/lon  → lat, lon

        Returns:
            DataFrame with columns:
                bay_id, status, zone_number, lat, lon
        """
        logger.info("Fetching live parking bay sensor data…")
        records = self._paginate(DATASET_LIVE_SENSORS)

        if not records:
            logger.warning("No live sensor records returned.")
            return pd.DataFrame(columns=["bay_id", "status", "zone_number", "lat", "lon"])

        # pd.json_normalize flattens nested dicts: location.lat, location.lon
        df = pd.json_normalize(records)

        rename = {
            "kerbsideid": "bay_id",
            "status_description": "status",
            "location.lat": "lat",
            "location.lon": "lon",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        keep = ["bay_id", "status", "zone_number", "lat", "lon"]
        available = [c for c in keep if c in df.columns]
        df = df[available].copy()

        logger.info("Live sensors fetched: %d rows", len(df))
        return df

    def get_parking_bays(self) -> pd.DataFrame:
        """Fetch GPS coordinates for all parking bays.

        Uses the CSV export endpoint (not the records endpoint) because the
        records endpoint rejects offset > 10,000 while the dataset has ~23,864
        rows.  The CSV is streamed into memory (~2 MB) and parsed with pandas.

        Live API schema (verified 2026-04):
            kerbsideid  → marker_id (join key to historical StreetMarker data)
            latitude    → lat
            longitude   → lon

        Returns:
            DataFrame with columns:
                marker_id, lat, lon
        """
        import io

        url = f"{self.base_url}/catalog/datasets/{DATASET_PARKING_BAYS}/exports/csv"
        logger.info("Fetching parking bays via CSV export from %s", url)

        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        # API returns UTF-8 with BOM; let pandas handle the encoding
        df = pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig", sep=";", low_memory=False)

        rename = {
            "kerbsideid": "marker_id",
            "roadsegmentdescription": "road_segment",
            "latitude": "lat",
            "longitude": "lon",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        keep = ["marker_id", "road_segment", "lat", "lon"]
        available = [c for c in keep if c in df.columns]
        df = df[available].copy()

        logger.info("Parking bays fetched: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _paginate(
        self,
        dataset_id: str,
        page_size: int = 100,
        extra_params: Optional[dict] = None,
    ) -> list[dict]:
        """Paginate through all records of a dataset.

        Args:
            dataset_id: Opendatasoft dataset identifier.
            page_size: Records per request (max 100 for this API).
            extra_params: Additional query parameters to include.

        Returns:
            List of record dicts (the ``results`` array from each page).
        """
        url = f"{self.base_url}/catalog/datasets/{dataset_id}/records"
        params: dict = {"limit": page_size, "offset": 0}
        if extra_params:
            params.update(extra_params)

        all_records: list[dict] = []
        page = 0

        while True:
            params["offset"] = page * page_size
            logger.debug(
                "GET %s offset=%d", dataset_id, params["offset"]
            )

            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code == 400:
                # Opendatasoft enforces a hard offset cap of 10,000; stop here.
                logger.warning(
                    "API returned 400 at offset=%d — offset limit reached, stopping pagination.",
                    params["offset"],
                )
                break
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            all_records.extend(results)

            total_count = data.get("total_count", 0)
            fetched_so_far = params["offset"] + len(results)

            logger.debug(
                "Fetched %d / %d records", fetched_so_far, total_count
            )

            if len(results) < page_size or fetched_so_far >= total_count:
                break

            page += 1
            # Polite delay to avoid hammering the API
            time.sleep(0.05)

        return all_records


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api = MelbourneAPI()

    # --- Test 1: live sensors (small, quick) ---
    print("\n=== Live Sensors ===")
    sensors_df = api.get_live_sensors()
    print(sensors_df.head())
    print(f"Shape: {sensors_df.shape}")
    if "status" in sensors_df.columns:
        print(f"Status counts:\n{sensors_df['status'].value_counts()}")

    # --- Test 2: parking bays GPS (small, quick) ---
    print("\n=== Parking Bays ===")
    bays_df = api.get_parking_bays()
    print(bays_df.head())
    print(f"Shape: {bays_df.shape}")

    # --- Test 3: historical CSV download (large — comment out to skip) ---
    # print("\n=== Historical CSV Download ===")
    # csv_path = api.download_historical_csv()
    # print(f"Saved to: {csv_path}")
