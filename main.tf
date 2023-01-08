# Configure the Google provider
provider "google" {
  version = "~> 3.34"
  project = "my-project-id"
  region  = "us-central1"
}

# Create a BigQuery dataset
resource "google_bigquery_dataset" "my_dataset" {
  dataset_id = "my_dataset"
}

# Create a BigQuery table
resource "google_bigquery_table" "my_table" {
  dataset_id    = google_bigquery_dataset.my_dataset.dataset_id
  table_id      = "my_table"
  time_partitioning {
    type = "DAY"
  }
}

# Create a Vertex AI Notebook instance
resource "google_ai_platform_notebook_instance" "my_instance" {
  name         = "my-instance"
  machine_type = "n1-standard-4"
  region       = "us-central1"

  # Attach the instance to the BigQuery dataset
  access_control {
    domain = "google.com"
    roles  = ["OWNER"]
  }
  service_account = "my-service-account@my-project-id.iam.gserviceaccount.com"
}
