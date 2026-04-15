export type Entity = {
  entity_id: string;
  name: string;
  label: string;
  aliases: string[];
  frequency: number;
  chunk_ids: string[];
};

export type SummaryResponse = {
  documents: number;
  chunks: number;
  entities: number;
  relations: number;
  top_entities: Entity[];
};

export type AskResponse = {
  answer: string;
  question: string;
  evidence: Array<{
    chunk_id: string;
    doc_id: string;
    score: number;
    text: string;
  }>;
  graph_context: Array<{
    source: string;
    target: string;
    relation: string;
    weight: number;
  }>;
};
