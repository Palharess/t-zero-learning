"""Export actual recorded W&B history from a completed offline run."""
import json
from pathlib import Path
import sys
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

folder = Path(sys.argv[1])
metadata = json.loads((folder / 'metadata.json').read_text())
source = Path(metadata['wandb_dir']).parent / ('run-' + metadata['wandb_id'] + '.wandb')
store = DataStore()
store.open_for_scan(str(source))
rows = []
while True:
    data = store.scan_data()
    if data is None:
        break
    record = wandb_internal_pb2.Record()
    record.ParseFromString(data)
    if record.HasField('history'):
        row = {}
        for item in record.history.item:
            key = item.key or '/'.join(item.nested_key)
            row[key] = json.loads(item.value_json)
        rows.append(row)
(folder / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
print(f'Exported {len(rows)} actual W&B history rows from {source.name}')
