from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def vdb_check_duplication(chunk, index, embeddings, threshold=0.99):
    try:
        duplicated = index.query(vector=embeddings.embed_query(chunk), top_k=1).get("matches")[0].get("score") >= threshold
    except IndexError:
        duplicated = False
    return chunk if not duplicated else None
    
def discard_duplicated_chunks(data_chunks, index, embeddings):
    empty_db = index.query(vector=embeddings.embed_query("test"), top_k=1).get("matches") == list()

    if not empty_db:
        deduplicated_data_chunks = list()
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(vdb_check_duplication, chunk, index, embeddings): chunk for chunk in data_chunks}

            for future in tqdm(as_completed(futures), total=len(data_chunks), desc="Validating duplicated data chunks against vector database"):
                result = future.result()
                if result:
                    deduplicated_data_chunks.append(result)
    else:
        deduplicated_data_chunks = data_chunks

    return deduplicated_data_chunks