import datetime
import hashlib
import io
from functools import lru_cache
from typing import Annotated, Any, TypeVar

import dateutil.parser
import torch
from cryptography.fernet import Fernet

__all__ = ["any2datetime", "any2date", "any2ymd", "Datetime"]

Datetime = TypeVar(
  "Datetime",
  Annotated[str, "string like '20220201', '2022-02-01' or 'today'."],
  datetime.datetime,
  datetime.date,
  Annotated[int, "int like 20220201."],
)


def any2datetime(dt: Datetime) -> datetime.datetime:
  if isinstance(dt, datetime.datetime):
    return dt
  elif isinstance(dt, datetime.date):
    return datetime.datetime.combine(dt, datetime.time())
  if isinstance(dt, int):
    return dateutil.parser.parse(str(dt))
  if isinstance(dt, str):
    if dt == "today":
      o = datetime.datetime.combine(datetime.datetime.now().date(), datetime.time())
    else:
      o = dateutil.parser.parse(dt)
    return o
  else:
    raise TypeError(f"dt type {type(dt)} not supported.")


def any2date(ts_input) -> datetime.date:
  return any2datetime(ts_input).date()


def any2ymd(ts_input) -> str:
  return any2date(ts_input).strftime("%Y-%m-%d")


class EncryptedCheckpointSaver:
  key = b"cQEifw8pZDVBQGRyGboYZeU6ZQshZsxRCnCiZaHKE1c="

  def __init__(self):
    self.fernet = Fernet(self.key)

  def save_encrypted_checkpoint(self, *, checkpoint: Any, filename: str):
    # Serialize the checkpoint
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    serialized_checkpoint = buffer.getvalue()

    # Encrypt the serialized checkpoint
    encrypted_checkpoint = self.fernet.encrypt(serialized_checkpoint)

    # Save the encrypted checkpoint
    with open(filename, "wb") as f:
      f.write(encrypted_checkpoint)

  def load_encrypted_checkpoint(self, filename: str) -> Any:
    # Read the encrypted checkpoint
    with open(filename, "rb") as f:
      encrypted_checkpoint = f.read()

    # Decrypt the checkpoint
    decrypted_checkpoint = self.fernet.decrypt(encrypted_checkpoint)

    # Deserialize the checkpoint
    buffer = io.BytesIO(decrypted_checkpoint)
    checkpoint = torch.load(buffer)

    return checkpoint


@lru_cache(maxsize=3000)
def anonymize(s: str) -> str:
  return hashlib.md5(s.encode()).hexdigest()
