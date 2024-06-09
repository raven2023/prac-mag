package com.example.speed_sheet3

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.RadioButton
import android.widget.RadioGroup
import androidx.appcompat.app.AppCompatActivity
import com.example.speed_sheet3.model.Endpoint

class SelectEndpointActivity : AppCompatActivity() {

    private val endpoints = listOf(
        Endpoint("Cell extraction + MNIST", "http://10.0.2.2:8000/"),
        Endpoint("Cell extraction + european dataset", "http://10.0.2.2:8001/"),
        Endpoint("Cell extraction + tesseract", "http://10.0.2.2:8003/"),
        Endpoint("Deepdoctection", "http://10.0.2.2:8004/"),
        Endpoint("ChatGPT-V(ision)", "http://10.0.2.2:8005/"),
        Endpoint("GoogleApiVision", "http://10.0.2.2:8006/"),
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_select_endpoint)



        val radioGroup = findViewById<RadioGroup>(R.id.radioGroupEndpoints)
        var firstRadioButtonId: Int = -1

        endpoints.forEachIndexed { index, endpoint ->
            val radioButton =(layoutInflater.inflate(R.layout.radio_button, radioGroup, false) as RadioButton).apply {
                id = View.generateViewId()
                text = endpoint.buttonName
                tag = endpoint.endpoint
                textSize = 24f
            }
            radioGroup.addView(radioButton)

            if (index == 0) {
                firstRadioButtonId = radioButton.id
            }
        }

        if (firstRadioButtonId != -1) {
            radioGroup.check(firstRadioButtonId)
        }

        val buttonSelectEndpoint = findViewById<Button>(R.id.buttonSelectEndpoint)
        buttonSelectEndpoint.setOnClickListener {
            val selectedId = radioGroup.checkedRadioButtonId
            val selectedEndpoint = endpoints[selectedId-1]

            val intent = Intent(this, MainActivity::class.java).apply {
                putExtra("selectedEndpoint", selectedEndpoint.endpoint)
            }
            startActivity(intent)
        }
    }
}
